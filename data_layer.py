"""
Data Layer for Pipeline Snapshot
Handles fetching data from MongoDB (req and application collections)

Based on: grading/algorithms/hs_algorithms/req_dynamics_creation/utils/data_layer.py
"""
from datetime import datetime
from typing import Dict, List, Optional
from toolz import get_in

from hs_gimme.db_facade.db_facade_factory import get_mongo_client_db
from hs_gimme.application_status_history_classifier.machine_learning_status_classifier import (
    get_machine_learning_status_classifier,
)
from hs_gimme.logging_service.logging_service import LoggingService
from hs_gimme.constants.objects.req import Req
from hs_gimme.constants.objects.application import Application, GradeData
from hs_gimme.constants.objects.ats_application import AtsApplication, AtsStatusInfo
from hs_gimme.constants.objects.ats_samurai import AtsSamurai

from constants import SnapshotFields


class PipelineDataLayer:
    """
    Handles all data fetching for pipeline snapshots.
    
    This class knows how to:
    1. Fetch req documents from MongoDB
    2. Fetch application documents for a req
    3. Parse status histories to get phase transitions
    """
    
    def __init__(self, account_id: str, environment: str):
        self.account_id = account_id
        self.environment = environment
        self.logger = LoggingService(prefix_constants=[("account_id", account_id)])
        
        # MongoDB client for direct queries
        self.mongo_client = get_mongo_client_db(
            account_id=account_id,
            environment=environment
        )
        
        # Status classifier - converts ATS status history to standardized phases
        self.status_classifier = get_machine_learning_status_classifier(
            account_id=account_id,
            env=environment
        )
    
    def get_req(self, req_id: str) -> Optional[Dict]:
        """
        Fetch a single req by ID.
        
        Returns:
            Dict with req data including open_date, req_status, status_history, etc.
            None if req not found
        """
        self.logger.info(f"Fetching req: {req_id}")
        
        req = self.mongo_client.req.find_one(
            {Req.ID: req_id},
            projection={
                Req.OPEN_DATE: 1,
                Req.REQ_STATUS: 1,
                Req.STATUS_HISTORY: 1,
                Req.CREATED_AT: 1,
            }
        )
        
        if not req:
            self.logger.warning(f"Req {req_id} not found")
            return None
        
        if not req.get(Req.OPEN_DATE):
            self.logger.warning(f"Req {req_id} has no open_date field")
        
        return req
    
    def get_applications_for_req(self, req_id: str) -> List[Dict]:
        """
        Fetch all applications for a req.
        
        Returns:
            List of application documents with:
            - application_id (_id)
            - req_id
            - grade_data.final_score
            - ats_application.status_info.status_history (for phase transitions)
            - ats_application.date_applied
            - ats_samurai.is_internal
            
        Note: Using ".".join() for nested paths to match req_dynamics pattern exactly.
        """
        self.logger.info(f"Fetching applications for req: {req_id}")
        
        applications = list(
            self.mongo_client.application.find(
                {Application.REQ_ID: req_id},
                projection={
                    Application.ID: 1,
                    Application.REQ_ID: 1,
                    ".".join([Application.GRADE_DATA, GradeData.FINAL_SCORE]): 1,
                    ".".join([
                        Application.ATS_APPLICATION,
                        AtsApplication.STATUS_INFO,
                        AtsStatusInfo.STATUS_HISTORY,
                    ]): 1,
                    ".".join([Application.ATS_APPLICATION, AtsApplication.DATE_APPLIED]): 1,
                    ".".join([Application.ATS_SAMURAI, AtsSamurai.IS_INTERNAL]): 1,
                }
            )
        )
        
        self.logger.info(f"Found {len(applications)} applications for req {req_id}")
        return applications
    
    def get_phases_reach_dates(self, application: Dict) -> Dict[str, datetime]:
        """
        Parse application's status_history to get when they reached each phase.
        
        This uses HiredScore's ML status classifier to convert ATS statuses
        (which vary by ATS system) into standardized phases.
        
        Args:
            application: Application document from MongoDB
            
        Returns:
            Dict mapping phase name to datetime when reached
            Example: {
                "Screening": datetime(2023, 1, 5),
                "Interview": datetime(2023, 1, 15),
                "Offer": datetime(2023, 1, 20)
            }
        """
        try:
            app_id = application.get('_id')
            
            # Debug: Check if status_history exists
            status_history = get_in(
                [Application.ATS_APPLICATION, AtsApplication.STATUS_INFO, AtsStatusInfo.STATUS_HISTORY],
                application
            )
            
            if not status_history:
                self.logger.warning(
                    f"Application {app_id} has no status_history",
                    application_keys=list(application.keys())
                )
                return {}
            
            self.logger.debug(
                f"Application {app_id} has status_history with {len(status_history)} items"
            )
            
            phases_dates = self.status_classifier.get_phases_reach_dates(application)
            
            if not phases_dates:
                self.logger.warning(
                    f"Status classifier returned empty phases for application {app_id}. "
                    f"Status history length: {len(status_history)}"
                )
            else:
                self.logger.debug(
                    f"Application {app_id} phases: {list(phases_dates.keys())}"
                )
            
            return phases_dates
            
        except Exception as e:
            self.logger.warning(
                f"Failed to get phases for application {application.get('_id')}: {e}",
                exc_info=True
            )
            return {}
    
    def get_application_grade(self, application: Dict) -> Optional[float]:
        """
        Extract grade (final_score) from application.
        
        Returns:
            Float grade (typically 0-5) or None if not graded
        """
        return get_in(
            [Application.GRADE_DATA, GradeData.FINAL_SCORE],
            application
        )
    
    def get_date_applied(self, application: Dict) -> Optional[datetime]:
        """
        Extract when candidate applied.
        
        Returns:
            Datetime of application or None
        """
        return get_in(
            [Application.ATS_APPLICATION, AtsApplication.DATE_APPLIED],
            application
        )
    
    def get_fill_date_from_status_history(self, req: Dict) -> Optional[datetime]:
        """
        Extract fill date from req status_history.
        
        The status_history structure is:
        {
            'Open': [datetime(...)],
            'Filled': [datetime(...)],
            'Closed': [datetime(...)],
        }
        
        Returns the first (earliest) Filled date if it exists.
        
        Args:
            req: Req document from MongoDB
            
        Returns:
            Datetime when req was filled, or None if not filled
        """
        status_history = req.get(Req.STATUS_HISTORY) or {}
        filled_dates = status_history.get('Filled', [])
        
        if filled_dates:
            # Return the first (earliest) filled date
            return sorted(filled_dates)[0]
        
        # Also check for 'Closed' status in case client uses that
        closed_dates = status_history.get('Closed', [])
        if closed_dates and req.get(Req.REQ_STATUS) in ['Filled', 'Closed']:
            return sorted(closed_dates)[0]
        
        return None
    
    def get_sample_filled_reqs(self, limit: int = 10) -> List[Dict]:
        """
        Get a sample of filled reqs for testing.
        
        Filters for:
        - Status = "Filled" (successfully hired someone)
        - Created after Nov 2022 (relatively recent data)
        - Has open_date set
        
        Args:
            limit: Max number of reqs to return
            
        Returns:
            List of req documents
        """
        self.logger.info(f"Fetching {limit} sample filled reqs")
        
        reqs = list(
            self.mongo_client.req.find(
                {
                    Req.REQ_STATUS: "Filled",
                    Req.CREATED_AT: {"$gt": datetime(2022, 11, 1)},
                    Req.OPEN_DATE: {"$exists": True}
                },
                projection={
                    Req.OPEN_DATE: 1,
                    Req.REQ_STATUS: 1,
                    Req.STATUS_HISTORY: 1,
                }
            ).limit(limit)
        )
        
        self.logger.info(f"Found {len(reqs)} filled reqs")
        return reqs

