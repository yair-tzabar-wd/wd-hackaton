"""
Pipeline Snapshot Builder - Core Component

This class answers: "What did the hiring pipeline look like on Day X?"

Given:
- A req_id
- Days since opening (e.g., day 5, day 12)

Returns:
- Count of candidates in each stage
- Grade distribution per stage
- Detailed application states

Based on: grading/algorithms/hs_algorithms/req_dynamics_creation/
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from hs_gimme.logging_service.logging_service import LoggingService
from hs_gimme.constants.objects.req import Req
from hs_algorithms.req_dynamics_creation.buckets_mapping import (
    get_inverse_phase_to_bucket_mapping,
)

from data_layer import PipelineDataLayer
from constants import (
    STAGE_TO_RANK,
    GRADE_LABELS,
    GRADE_BOUNDARIES,
    SnapshotFields,
)


class PipelineSnapshot:
    """
    Builds a snapshot of the hiring pipeline at a specific point in time.
    
    This is the core "Timeline Engineer" component.
    """
    
    def __init__(self, account_id: str, environment: str):
        """
        Initialize the snapshot builder.
        
        Args:
            account_id: HiredScore account ID
            environment: "production", "staging", etc.
        """
        self.account_id = account_id
        self.environment = environment
        self.logger = LoggingService(prefix_constants=[("account_id", account_id)])
        
        # Data layer handles all MongoDB queries
        self.data_layer = PipelineDataLayer(account_id, environment)
        
        # Phase bucket mapping - maps ATS-specific phase names to standard buckets
        self.phase_bucket_mapping = get_inverse_phase_to_bucket_mapping(account_id)
        
        self.logger.info("PipelineSnapshot initialized")
    
    def get_snapshot(self, req_id: str, days_since_open: int) -> Dict:
        """
        Get the pipeline state at a specific day.
        
        This is the main entry point. It orchestrates the entire process:
        1. Fetch req data
        2. Calculate target date
        3. Fetch all applications
        4. Determine each application's state at target date
        5. Aggregate into counts
        
        Args:
            req_id: The requisition ID
            days_since_open: How many days after opening (0 = opening day)
            
        Returns:
            Dict with snapshot data:
            {
                'req_id': 'req_12345',
                'target_date': datetime(...),
                'days_since_open': 5,
                'total_active': 12,
                'Screening_all': 8,
                'Screening_A': 3,
                'Interview_all': 2,
                ...
            }
        """
        self.logger.info(
            f"Building snapshot for req {req_id} at day {days_since_open}"
        )
        
        # STEP 1: Get req data (especially open_date)
        req = self.data_layer.get_req(req_id)
        if not req:
            raise ValueError(f"Req {req_id} not found")
        
        date_open = req.get(Req.OPEN_DATE)
        if not date_open:
            raise ValueError(f"Req {req_id} has no open_date")
        
        # STEP 2: Calculate target date (the day we're "rewinding" to)
        target_date = date_open + timedelta(days=days_since_open)
        self.logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        
        # STEP 3: Get all applications for this req
        applications = self.data_layer.get_applications_for_req(req_id)
        self.logger.info(f"Processing {len(applications)} applications")
        
        # STEP 4: For each application, determine its state at target_date
        application_states = []
        for app in applications:
            app_state = self._get_application_state_at_date(
                app, target_date, date_open
            )
            if app_state:  # Only include if they were active
                application_states.append(app_state)
        
        self.logger.info(
            f"Found {len(application_states)} active applications at target date"
        )
        
        # STEP 5: Aggregate into counts (like "Screening_A: 3")
        snapshot = self._aggregate_snapshot(
            application_states,
            target_date,
            days_since_open,
            req_id
        )
        
        return snapshot
    
    def _get_application_state_at_date(
        self,
        application: Dict,
        target_date: datetime,
        req_date_open: datetime
    ) -> Optional[Dict]:
        """
        Determine what stage an application was in at a specific date.
        
        This is the CORE LOGIC - figuring out "where was this candidate on Day X?"
        
        Process:
        1. Check if they had applied by target_date (if not, skip them)
        2. Get all phases they reached and when
        3. Find the most recent phase that happened before/on target_date
        4. Return their state (stage, grade, etc.)
        
        Args:
            application: Application document from MongoDB
            target_date: The date we're checking
            req_date_open: When the req opened
            
        Returns:
            Dict with application state or None if not active yet:
            {
                'application_id': 'app_123',
                'stage': 'Screening',
                'grade': 4.2,
                'grade_bucket': 'A',
                'stage_entered_date': datetime(...),
                'days_in_stage': 3
            }
        """
        app_id = application.get('_id')
        
        # Get when candidate applied
        date_applied = self.data_layer.get_date_applied(application)
        
        # If they applied after target_date, they weren't in pipeline yet
        if date_applied and date_applied > target_date:
            return None
        
        # Get all phases this candidate reached (with timestamps)
        phases_dates = self.data_layer.get_phases_reach_dates(application)
        
        if not phases_dates:
            # No phase data - treat as "just applied"
            if date_applied and date_applied <= target_date:
                return self._create_application_state(
                    app_id, "Screening", date_applied, target_date, application
                )
            return None
        
        # Map phases to standard buckets (e.g., "Phone Screen" -> "Screening")
        phases_dates_mapped = {
            self.phase_bucket_mapping.get(phase, phase): date
            for phase, date in phases_dates.items()
        }
        
        # Filter to only standard stages we care about
        valid_phases = {
            phase: date
            for phase, date in phases_dates_mapped.items()
            if phase in STAGE_TO_RANK.keys()
        }
        
        if not valid_phases:
            # Has phases but none are standard - treat as screening
            if date_applied and date_applied <= target_date:
                return self._create_application_state(
                    app_id, "Screening", date_applied, target_date, application
                )
            return None
        
        # Find the most recent stage that happened by target_date
        current_stage = None
        current_stage_date = None
        
        for phase, phase_date in sorted(valid_phases.items(), key=lambda x: x[1]):
            if phase_date <= target_date:
                # This phase happened by target_date
                current_stage = phase
                current_stage_date = phase_date
            else:
                # This phase happened AFTER target_date, stop looking
                break
        
        if not current_stage:
            # No phases reached by target_date, but they had applied
            if date_applied and date_applied <= target_date:
                return self._create_application_state(
                    app_id, "Screening", date_applied, target_date, application
                )
            return None
        
        # Create the state dict
        return self._create_application_state(
            app_id, current_stage, current_stage_date, target_date, application
        )
    
    def _create_application_state(
        self,
        app_id: str,
        stage: str,
        stage_date: datetime,
        target_date: datetime,
        application: Dict
    ) -> Dict:
        """
        Create a standardized application state dict.
        
        Args:
            app_id: Application ID
            stage: Current stage name
            stage_date: When they entered this stage
            target_date: The date we're building snapshot for
            application: Full application document (to extract grade)
            
        Returns:
            Dict with standardized fields
        """
        grade = self.data_layer.get_application_grade(application)
        grade_bucket = self._get_grade_bucket(grade)
        days_in_stage = (target_date - stage_date).days if stage_date else 0
        
        return {
            SnapshotFields.APPLICATION_ID: app_id,
            SnapshotFields.STAGE: stage,
            SnapshotFields.GRADE: grade,
            SnapshotFields.GRADE_BUCKET: grade_bucket,
            SnapshotFields.STAGE_ENTERED_DATE: stage_date,
            SnapshotFields.DAYS_IN_STAGE: days_in_stage,
        }
    
    def _get_grade_bucket(self, grade: Optional[float]) -> str:
        """
        Map numeric grade to letter bucket (A/B/C/D/N).
        
        Buckets:
        - N (None): No grade or <= 0
        - D (Bad): 0-2
        - C (Okay): 2-3
        - B (Good): 3-4
        - A (Excellent): 4-5
        
        Args:
            grade: Numeric grade (0-5 scale) or None
            
        Returns:
            Letter grade bucket
        """
        if grade is None or grade <= 0:
            return 'N'
        elif grade <= 2:
            return 'D'
        elif grade <= 3:
            return 'C'
        elif grade <= 4:
            return 'B'
        else:
            return 'A'
    
    def _aggregate_snapshot(
        self,
        application_states: List[Dict],
        target_date: datetime,
        days_since_open: int,
        req_id: str
    ) -> Dict:
        """
        Aggregate individual application states into pipeline-level counts.
        
        Converts a list like:
        [
            {'stage': 'Screening', 'grade_bucket': 'A'},
            {'stage': 'Screening', 'grade_bucket': 'B'},
            {'stage': 'Interview', 'grade_bucket': 'A'},
        ]
        
        Into counts like:
        {
            'Screening_A': 1,
            'Screening_B': 1,
            'Screening_all': 2,
            'Interview_A': 1,
            'Interview_all': 1,
            'total_active': 3
        }
        
        Args:
            application_states: List of application state dicts
            target_date: The date of this snapshot
            days_since_open: Days since req opened
            req_id: Requisition ID
            
        Returns:
            Dict with aggregated counts
        """
        df = pd.DataFrame(application_states) if application_states else pd.DataFrame()
        
        result = {
            SnapshotFields.REQ_ID: req_id,
            SnapshotFields.TARGET_DATE: target_date,
            SnapshotFields.DAYS_SINCE_OPEN: days_since_open,
            SnapshotFields.TOTAL_ACTIVE: len(application_states),
        }
        
        if len(df) == 0:
            # Empty pipeline - fill with zeros
            for stage in STAGE_TO_RANK.keys():
                result[f'{stage}_all'] = 0
                result[f'{stage}_good'] = 0
                result[f'{stage}_moderate'] = 0
                for grade in GRADE_LABELS:
                    result[f'{stage}_{grade}'] = 0
            result[SnapshotFields.APPLICATIONS_DETAIL] = []
            return result
        
        # Count by stage + grade (e.g., "Screening_A": 3)
        stage_grade_counts = df.groupby([SnapshotFields.STAGE, SnapshotFields.GRADE_BUCKET]).size()
        
        for stage in STAGE_TO_RANK.keys():
            # Individual grade counts
            for grade in GRADE_LABELS:
                key = (stage, grade)
                result[f'{stage}_{grade}'] = int(stage_grade_counts.get(key, 0))
            
            # Aggregated counts (following req_dynamics pattern)
            result[f'{stage}_good'] = result[f'{stage}_A'] + result[f'{stage}_B']
            result[f'{stage}_moderate'] = (
                result[f'{stage}_C'] +
                result[f'{stage}_D'] +
                result[f'{stage}_N']
            )
            result[f'{stage}_all'] = result[f'{stage}_good'] + result[f'{stage}_moderate']
        
        # Include detailed application list for debugging
        result[SnapshotFields.APPLICATIONS_DETAIL] = application_states
        
        return result

