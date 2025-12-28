import os

os.environ['HIREDSCORE_CELL'] = "0000"
os.environ['SD_MONGODB_ATLAS_HIREDSCORE_APPLICATIVE_SECRET_ENABLED_PREPROD_0000'] = "true"
os.environ['SD_MONGODB_ATLAS_HIREDSCORE_APPLICATIVE_SECRET_HOST_PREPROD_0000'] = "preprod-0000-applicative-pl-0.jfiwu5.mongodb.net"
os.environ['SD_MONGODB_ATLAS_HIREDSCORE_APPLICATIVE_SECRET_USERNAME_PREPROD_0000'] = "yairtzabar_apono-preprod-0000-applicative"
os.environ['SD_MONGODB_ATLAS_HIREDSCORE_APPLICATIVE_SECRET_PASSWORD_PREPROD_0000'] = "5l;^QuS4&7*fUrV1"

import logging
from typing import Optional, Dict, Any

from pymongo.database import Database

from hs_gimme.db_facade.db_facade_factory import get_mongo_client_db
from hs_gimme.logging_service.logging_service import LoggingService
from hs_gimme.constants.objects.req import Req
from hs_gimme.constants.objects.application import Application

logger = LoggingService(logging.getLogger(__name__))


# =============================================================================
# Task 2.1: Extract Static Features from Requisition
# =============================================================================

def extract_req_static_features(req_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract static features from a requisition (req) JSON document.
    These features indicate how hard the role is to fill.
    """
    if not req_doc:
        return {
            "req_id": None,
            "is_internal": None,
            "seniority": None,
            "top_profession": None,
            "avg_skill_score": None,
        }
    
    req_id = req_doc.get("_id")
    
    is_internal = req_doc.get(Req.IS_INTERNAL)
    if is_internal is None:
        is_internal = req_doc.get("is_internal_req", False)
    
    seniority = _extract_seniority(req_doc)
    top_profession = _extract_top_profession(req_doc)
    avg_skill_score = _extract_avg_skill_score(req_doc)
    
    return {
        "req_id": req_id,
        "is_internal": bool(is_internal) if is_internal is not None else False,
        "seniority": seniority,
        "top_profession": top_profession,
        "avg_skill_score": avg_skill_score,
    }


def _extract_seniority(req_doc: Dict[str, Any]) -> str:
    """Extract and normalize seniority level from requisition."""
    seniority = req_doc.get(Req.SENIORITY_LEVEL)
    
    if not seniority:
        seniority = req_doc.get(Req.JOB_LEVEL)
    
    if not seniority:
        job_band = req_doc.get(Req.EXTERNAL_JOB_BAND) or req_doc.get(Req.CALCULATED_JOB_BAND)
        if job_band:
            seniority = _job_band_to_seniority(job_band)
    
    if not seniority:
        seniority = req_doc.get(Req.MANAGEMENT_LEVEL)
    
    return _normalize_seniority(seniority) if seniority else "Unknown"


def _normalize_seniority(value: Any) -> str:
    """Normalize seniority value to standard levels."""
    if value is None:
        return "Unknown"
    
    if isinstance(value, dict):
        value = list(value.values())[0] if value else None
    
    if not isinstance(value, str):
        return "Unknown"
    
    value_lower = value.lower().strip()
    
    if any(x in value_lower for x in ["entry", "intern", "trainee", "graduate"]):
        return "Entry"
    elif any(x in value_lower for x in ["junior", "jr", "associate", "i ", " i"]):
        return "Junior"
    elif any(x in value_lower for x in ["mid", "intermediate", "ii ", " ii"]):
        return "Mid"
    elif any(x in value_lower for x in ["senior", "sr", "iii", "lead", "principal"]):
        return "Senior"
    elif any(x in value_lower for x in ["director", "vp", "executive", "chief", "head"]):
        return "Executive"
    elif any(x in value_lower for x in ["manager", "mgr"]):
        return "Manager"
    
    return value.title()


def _job_band_to_seniority(job_band: Any) -> str:
    """Convert job band to seniority level."""
    if job_band is None:
        return None
    
    band_str = str(job_band).lower()
    
    if any(x in band_str for x in ["1", "entry", "intern"]):
        return "Entry"
    elif any(x in band_str for x in ["2", "junior"]):
        return "Junior"
    elif any(x in band_str for x in ["3", "mid"]):
        return "Mid"
    elif any(x in band_str for x in ["4", "5", "senior"]):
        return "Senior"
    elif any(x in band_str for x in ["6", "7", "8", "director", "executive"]):
        return "Executive"
    
    return None


def _extract_top_profession(req_doc: Dict[str, Any]) -> str:
    """Extract top profession/category from requisition."""
    top_profession = req_doc.get(Req.TOP_CATEGORY)
    
    if not top_profession:
        categories = req_doc.get(Req.CATEGORIES, [])
        if categories and isinstance(categories, list) and len(categories) > 0:
            first_cat = categories[0]
            if isinstance(first_cat, dict):
                top_profession = first_cat.get("top_category") or first_cat.get("name")
            elif isinstance(first_cat, str):
                top_profession = first_cat
    
    if not top_profession:
        top_profession = req_doc.get(Req.JOB_FAMILY)
    
    if not top_profession:
        top_profession = req_doc.get(Req.JOB_FUNCTION)
    
    if not top_profession:
        profession_scores = req_doc.get(Req.PROFESSION_SCORES, {})
        if profession_scores and isinstance(profession_scores, dict):
            try:
                top_profession = max(profession_scores.items(), key=lambda x: x[1])[0]
            except (ValueError, TypeError):
                pass
    
    return top_profession or "Unknown"


def _extract_avg_skill_score(req_doc: Dict[str, Any]) -> Optional[float]:
    """Extract average skill score from requisition skills."""
    skill_scores = []
    
    skills = req_doc.get(Req.CALCULATED_REQUIRED_SKILLS, [])
    if skills and isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                score = skill.get("importance") or skill.get("score") or skill.get("skill_score")
                if score is not None and isinstance(score, (int, float)):
                    skill_scores.append(float(score))
    
    if not skill_scores:
        skills = req_doc.get(Req.SKILLS, [])
        if skills and isinstance(skills, list):
            for skill in skills:
                if isinstance(skill, dict):
                    score = skill.get("importance") or skill.get("score")
                    if score is not None and isinstance(score, (int, float)):
                        skill_scores.append(float(score))
    
    if not skill_scores:
        kb_skills = req_doc.get(Req.KNOWLEDGE_BASE_SKILLS, {})
        if isinstance(kb_skills, dict):
            skill_items = kb_skills.get("skill_items_from_kb", [])
            for skill in skill_items:
                if isinstance(skill, dict):
                    score = skill.get("skill_probability")
                    if score is not None and isinstance(score, (int, float)):
                        skill_scores.append(float(score))
    
    if skill_scores:
        return round(sum(skill_scores) / len(skill_scores), 4)
    
    return None


# =============================================================================
# ReqDataFetcher Class
# =============================================================================

class ReqDataFetcher:
    """Client for fetching requisition data from MongoDB."""

    REQ_COLLECTION = "req"
    APPLICATION_COLLECTION = "application"

    def __init__(self, environment: str, account_id: str):
        self.environment = environment
        self.account_id = account_id
        self._db: Optional[Database] = None

    @property
    def db(self) -> Database:
        """Get the applicative MongoDB database connection."""
        if self._db is None:
            self._db = get_mongo_client_db(
                environment=self.environment,
                account_id=self.account_id,
                connect=True
            )
        return self._db

    @property
    def req_collection(self):
        """Get the requisitions collection."""
        return self.db[self.REQ_COLLECTION]

    @property
    def application_collection(self):
        """Get the applications collection."""
        return self.db[self.APPLICATION_COLLECTION]

    def get_req_by_id(self, req_id: str) -> Optional[Dict]:
        """Get a single requisition by ID."""
        return self.req_collection.find_one({"_id": req_id})

    # ==========================================================================
    # Task 2.2: Success Calculator - The "Answer Key"
    # ==========================================================================

    def count_unique_applicants_for_req(self, req_id: str) -> int:
        """Count the number of unique candidates who applied to a requisition."""
        pipeline = [
            {"$match": {Application.REQ_ID: req_id}},
            {"$group": {"_id": f"${Application.CANDIDATE_ID}"}},
            {"$count": "unique_applicants"}
        ]
        result = list(self.application_collection.aggregate(pipeline))
        return result[0]["unique_applicants"] if result else 0

    def get_success_metrics_for_req(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        Task 2.2: Get the "Magic Number" for a specific requisition.
        
        Returns:
            - req_id: The requisition ID
            - is_successful: Whether the req was successfully filled
            - final_total_applicants: Count of unique humans who applied
            - status: The requisition status
            - fill_date: When the position was filled
        """
        req = self.get_req_by_id(req_id)
        if not req:
            return None
        
        status = req.get(Req.REQ_STATUS, "Unknown")
        fill_date = req.get(Req.FILL_DATE)
        
        successful_statuses = {"Closed", "Filled", "Closed - Filled"}
        is_successful = status in successful_statuses or fill_date is not None
        
        unique_applicants = self.count_unique_applicants_for_req(req_id)
        
        return {
            "req_id": req_id,
            "is_successful": is_successful,
            "final_total_applicants": unique_applicants,
            "status": status,
            "fill_date": fill_date,
        }


def create_req_fetcher(environment: str, account_id: str) -> ReqDataFetcher:
    return ReqDataFetcher(environment, account_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    ENVIRONMENT = "preprod" 
    ACCOUNT_ID = "utah"
    REQ_ID = "UNLR-24679"

    fetcher = create_req_fetcher(ENVIRONMENT, ACCOUNT_ID)

    req = fetcher.get_req_by_id(REQ_ID)
    
    if req:
        static_features = extract_req_static_features(req)
        success_metrics = fetcher.get_success_metrics_for_req(REQ_ID)
