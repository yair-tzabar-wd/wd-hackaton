"""
Constants for Pipeline Snapshot Builder
Based on req_dynamics_creation/utils/constants.py

Uses actual field names from hs_gimme.constants.objects
"""
from hs_gimme.constants.objects.req import Req
from hs_gimme.constants.objects.application import Application, GradeData
from hs_gimme.constants.objects.ats_application import AtsApplication, AtsStatusInfo
from hs_gimme.constants.objects.ats_samurai import AtsSamurai

# Stage to rank mapping - defines the standard pipeline stages
STAGE_TO_RANK = {
    "Screening": 1,
    "Hiring Manager Review": 2,
    "Interview": 3,
    "Offer": 4,
    "Rejected": 4,
}

# Grade buckets - how we categorize candidate quality
GRADE_LABELS = ["N", "D", "C", "B", "A"]

# Grade boundaries
# N = No grade or <= 0
# D = 0-2
# C = 2-3
# B = 3-4
# A = 4-5
GRADE_BOUNDARIES = [-2, 0, 2, 3, 4, 5]


class SnapshotFields:
    """Field names in snapshot output"""
    REQ_ID = "req_id"
    TARGET_DATE = "target_date"
    DAYS_SINCE_OPEN = "days_since_open"
    TOTAL_ACTIVE = "total_active"
    APPLICATIONS_DETAIL = "applications_detail"
    
    # Per-application fields
    APPLICATION_ID = "application_id"
    STAGE = "stage"
    GRADE = "grade"
    GRADE_BUCKET = "grade_bucket"
    STAGE_ENTERED_DATE = "stage_entered_date"
    DAYS_IN_STAGE = "days_in_stage"

