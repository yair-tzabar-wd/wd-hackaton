"""
Static Feature Engine for Hiring Pipeline Predictor

This module extracts "static" features from Job Application JSON documents.
These features represent the difficulty/context of filling a role and don't change over time.

Developer 2 - Context Builder
"""

import logging
from typing import Dict, Any, Optional, List, Union
from statistics import mean

from hs_gimme.logging_service.logging_service import LoggingService

# HiredScore constants - using proper gimme package classes
from hs_gimme.constants.objects.application import Application
from hs_gimme.constants.objects.ats_application import AtsApplication
from hs_gimme.constants.objects.samurai_json import (
    SamuraiJson,
    SamuraiExperienceItem,
    SkillsKB,
    SkillsKBItem,
)

logger = LoggingService(logging.getLogger(__name__))


class SeniorityLevels:
    """Standard seniority level mappings."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

    # Normalized ordering for ML features
    LEVEL_ORDER = {
        ENTRY: 1,
        JUNIOR: 2,
        MID: 3,
        SENIOR: 4,
        LEAD: 5,
        EXECUTIVE: 6,
        UNKNOWN: 0
    }


# =============================================================================
# HELPER FUNCTIONS - Safe data extraction
# =============================================================================

def safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """
    Safely navigate nested dictionary keys.
    
    Args:
        data: The dictionary to navigate
        *keys: Sequence of keys to traverse
        default: Default value if path doesn't exist
        
    Returns:
        The value at the path or default
    """
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
        if result is None:
            return default
    return result


def safe_get_first(items: Union[List, Dict, None], default: Any = None) -> Any:
    """
    Get the first item from a list or dict values.
    
    Args:
        items: List, dict, or None
        default: Default value if empty
        
    Returns:
        First item or default
    """
    if items is None:
        return default
    
    if isinstance(items, dict):
        values = list(items.values())
        return values[0] if values else default
    
    if isinstance(items, list):
        return items[0] if items else default
    
    return items


def normalize_seniority(seniority_value: Any) -> str:
    """
    Normalize seniority value to standard levels.
    
    Args:
        seniority_value: Raw seniority value (str, dict, or None)
        
    Returns:
        Normalized seniority string
    """
    if seniority_value is None:
        return SeniorityLevels.UNKNOWN
    
    # If it's a dict like {'Medical': 'junior'}, extract the value
    if isinstance(seniority_value, dict):
        seniority_value = safe_get_first(seniority_value)
    
    if not isinstance(seniority_value, str):
        return SeniorityLevels.UNKNOWN
    
    seniority_lower = seniority_value.lower().strip()
    
    # Map common variations to standard levels
    seniority_mapping = {
        "entry": SeniorityLevels.ENTRY,
        "entry-level": SeniorityLevels.ENTRY,
        "entry level": SeniorityLevels.ENTRY,
        "junior": SeniorityLevels.JUNIOR,
        "jr": SeniorityLevels.JUNIOR,
        "mid": SeniorityLevels.MID,
        "mid-level": SeniorityLevels.MID,
        "mid level": SeniorityLevels.MID,
        "intermediate": SeniorityLevels.MID,
        "senior": SeniorityLevels.SENIOR,
        "sr": SeniorityLevels.SENIOR,
        "lead": SeniorityLevels.LEAD,
        "principal": SeniorityLevels.LEAD,
        "staff": SeniorityLevels.LEAD,
        "executive": SeniorityLevels.EXECUTIVE,
        "director": SeniorityLevels.EXECUTIVE,
        "vp": SeniorityLevels.EXECUTIVE,
        "c-level": SeniorityLevels.EXECUTIVE,
    }
    
    return seniority_mapping.get(seniority_lower, SeniorityLevels.UNKNOWN)


# =============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_static_features(json_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract static features from a single Job Application JSON document.
    Uses constants from hs_gimme (Application, AtsApplication, SamuraiJson, etc.)
    
    These features represent the "difficulty" or context of filling a role.
    They don't change over the lifecycle of the requisition.
    
    Args:
        json_doc: A single application JSON document from MongoDB
        
    Returns:
        Dictionary containing extracted features:
            - req_id: The requisition ID
            - candidate_id: The candidate ID
            - is_internal: 1 if internal candidate, 0 if external
            - seniority_level: Normalized seniority (e.g., 'junior', 'senior')
            - seniority_numeric: Numeric encoding of seniority (1-6)
            - profession_group: Primary profession category
            - avg_candidate_quality: Mean skill score for the candidate
            - skill_count: Number of skills identified
    """
    try:
        features = {}
        
        # =====================================================================
        # 1. Basic IDs - using Application constants from hs_gimme
        # =====================================================================
        features["req_id"] = json_doc.get(Application.REQ_ID)
        features["candidate_id"] = json_doc.get(Application.CANDIDATE_ID)
        
        # =====================================================================
        # 2. Is Internal Flag - using AtsApplication constants
        # =====================================================================
        is_internal = None
        
        # Try ats_application.internal_candidate_info using AtsApplication constants
        ats_app = json_doc.get(Application.ATS_APPLICATION, {}) or {}
        
        # Check internal_candidate_info for is_internal_candidate
        internal_info = ats_app.get(AtsApplication.INTERNAL_CANDIDATE_INFO, {}) or {}
        is_internal = internal_info.get("is_internal_candidate")
        
        # Also check samurai_json.is_internal using SamuraiJson constant
        if is_internal is None:
            samurai_json = json_doc.get("samurai_json", {}) or {}
            is_internal = samurai_json.get(SamuraiJson.IS_INTERNAL)
        
        # Also check top-level is_internal_candidate if present
        if is_internal is None:
            is_internal = json_doc.get("is_internal_candidate")
        
        features["is_internal"] = 1 if is_internal else 0
        
        # =====================================================================
        # 3. Seniority Level - using SamuraiJson and SamuraiExperienceItem constants
        # =====================================================================
        seniority_raw = None
        
        # Try ats_samurai.experience.seniority path
        ats_samurai = json_doc.get(Application.ATS_SAMURAI, {}) or {}
        experience = ats_samurai.get(SamuraiJson.EXPERIENCE)
        
        if experience:
            if isinstance(experience, dict):
                seniority_raw = experience.get(SamuraiExperienceItem.SENIORITY)
            elif isinstance(experience, list) and experience:
                # If experience is a list, get seniority from most recent/first item
                first_exp = experience[0] if experience else {}
                seniority_raw = first_exp.get(SamuraiExperienceItem.SENIORITY)
        
        # Also try samurai_json.seniority (top-level)
        if seniority_raw is None:
            samurai_json = json_doc.get("samurai_json", {}) or {}
            seniority_raw = samurai_json.get(SamuraiJson.SENIORITY)
            
            # Fallback to experience items
            if seniority_raw is None:
                exp_data = samurai_json.get(SamuraiJson.EXPERIENCE, {}) or {}
                if isinstance(exp_data, dict):
                    seniority_raw = exp_data.get(SamuraiExperienceItem.SENIORITY)
                elif isinstance(exp_data, list) and exp_data:
                    seniority_raw = exp_data[0].get(SamuraiExperienceItem.SENIORITY) if exp_data else None
        
        features["seniority_level"] = normalize_seniority(seniority_raw)
        features["seniority_numeric"] = SeniorityLevels.LEVEL_ORDER.get(
            features["seniority_level"], 0
        )
        
        # =====================================================================
        # 4. Top Profession / Profession Group - using SamuraiJson constants
        # =====================================================================
        profession_group = None
        
        # Try samurai_json.professions first using SamuraiJson constant
        samurai_json = json_doc.get("samurai_json", {}) or {}
        professions = samurai_json.get(SamuraiJson.PROFESSIONS)
        if professions:
            profession_group = _extract_top_profession(professions)
        
        # Fallback to experience.top_professions
        if profession_group is None and experience:
            if isinstance(experience, dict):
                top_profs = experience.get("top_professions")
                profession_group = _extract_top_profession(top_profs)
            elif isinstance(experience, list) and experience:
                first_exp = experience[0] if experience else {}
                top_profs = first_exp.get("top_professions")
                profession_group = _extract_top_profession(top_profs)
        
        features["profession_group"] = profession_group or "Unknown"
        
        # =====================================================================
        # 5. Average Candidate Quality (Skill Scores) - using SkillsKB constants
        # =====================================================================
        skill_scores = []
        
        # Try samurai_json.skills_kb.skills_kb_items using SkillsKB constants
        samurai_json = json_doc.get("samurai_json", {}) or {}
        skills_kb = samurai_json.get(SamuraiJson.SKILLS_KB, {}) or {}
        skills_kb_items = skills_kb.get(SkillsKB.SKILLS_KB_ITEMS, []) or []
        
        for skill_item in skills_kb_items:
            if isinstance(skill_item, dict):
                score = skill_item.get(SkillsKBItem.SKILL_SCORE)
                if score is not None and isinstance(score, (int, float)):
                    skill_scores.append(float(score))
        
        # Also try ats_samurai.skills_kb.skills_kb_items as fallback
        if not skill_scores:
            ats_samurai_skills = ats_samurai.get(SamuraiJson.SKILLS_KB, {}) or {}
            skills_items = ats_samurai_skills.get(SkillsKB.SKILLS_KB_ITEMS, []) or []
            for skill_item in skills_items:
                if isinstance(skill_item, dict):
                    score = skill_item.get(SkillsKBItem.SKILL_SCORE)
                    if score is not None and isinstance(score, (int, float)):
                        skill_scores.append(float(score))
        
        features["avg_candidate_quality"] = round(mean(skill_scores), 4) if skill_scores else None
        features["skill_count"] = len(skill_scores)
        
        return features
        
    except Exception as e:
        logger.error(
            "Error extracting static features",
            error=str(e),
            doc_id=json_doc.get("_id")
        )
        # Return minimal features on error
        return {
            "req_id": json_doc.get(Application.REQ_ID),
            "candidate_id": json_doc.get(Application.CANDIDATE_ID),
            "is_internal": 0,
            "seniority_level": SeniorityLevels.UNKNOWN,
            "seniority_numeric": 0,
            "profession_group": "Unknown",
            "avg_candidate_quality": None,
            "skill_count": 0,
            "_extraction_error": str(e)
        }


def _extract_top_profession(top_professions: Any) -> Optional[str]:
    """
    Extract the highest-scoring profession from top_professions data.
    
    Args:
        top_professions: Nested list/dict of profession data
        
    Returns:
        Name of the top profession or None
    """
    if top_professions is None:
        return None
    
    # Handle different data structures
    if isinstance(top_professions, str):
        return top_professions
    
    if isinstance(top_professions, dict):
        # Could be like {'IT_&_Software_Development': 0.95, 'Engineering': 0.80}
        if top_professions:
            # Return the one with highest score
            try:
                return max(top_professions.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]
            except (ValueError, TypeError):
                return list(top_professions.keys())[0]
        return None
    
    if isinstance(top_professions, list):
        if not top_professions:
            return None
        
        first_item = top_professions[0]
        
        # Could be a list of dicts like [{'name': 'Engineering', 'score': 0.9}]
        if isinstance(first_item, dict):
            # Look for name/entity/profession key
            for key in ['name', 'profession', 'entity', 'skill_entity']:
                if key in first_item:
                    return first_item[key]
            # Just return first value if no known key
            if first_item:
                return str(list(first_item.values())[0])
        
        # Could be simple list of strings
        if isinstance(first_item, str):
            return first_item
        
        # Nested list
        if isinstance(first_item, list) and first_item:
            nested = first_item[0]
            if isinstance(nested, str):
                return nested
            if isinstance(nested, dict):
                for key in ['name', 'profession', 'entity']:
                    if key in nested:
                        return nested[key]
    
    return None


# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def extract_features_batch(
    documents: List[Dict[str, Any]],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Extract static features from a batch of application documents.
    
    Args:
        documents: List of application JSON documents
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of feature dictionaries
    """
    features_list = []
    total = len(documents)
    
    for i, doc in enumerate(documents):
        features = extract_static_features(doc)
        features_list.append(features)
        
        if progress_callback and (i + 1) % 1000 == 0:
            progress_callback(i + 1, total)
    
    logger.info(
        "Batch feature extraction complete",
        total_documents=total,
        successful=len([f for f in features_list if "_extraction_error" not in f])
    )
    
    return features_list


def aggregate_features_by_req(
    features_list: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate candidate-level features to requisition level.
    
    This creates summary statistics per requisition that can be used
    as model features.
    
    Args:
        features_list: List of candidate-level feature dicts
        
    Returns:
        Dictionary mapping req_id to aggregated features
    """
    from collections import defaultdict
    
    req_data = defaultdict(lambda: {
        "internal_count": 0,
        "external_count": 0,
        "seniority_levels": [],
        "profession_groups": [],
        "quality_scores": [],
        "skill_counts": [],
        "candidate_ids": set()
    })
    
    for features in features_list:
        req_id = features.get("req_id")
        if not req_id:
            continue
            
        data = req_data[req_id]
        
        # Count internal vs external
        if features.get("is_internal") == 1:
            data["internal_count"] += 1
        else:
            data["external_count"] += 1
        
        # Collect seniority levels
        if features.get("seniority_level"):
            data["seniority_levels"].append(features["seniority_level"])
        
        # Collect profession groups
        if features.get("profession_group") and features["profession_group"] != "Unknown":
            data["profession_groups"].append(features["profession_group"])
        
        # Collect quality scores
        if features.get("avg_candidate_quality") is not None:
            data["quality_scores"].append(features["avg_candidate_quality"])
        
        # Collect skill counts
        if features.get("skill_count"):
            data["skill_counts"].append(features["skill_count"])
        
        # Track unique candidates
        if features.get("candidate_id"):
            data["candidate_ids"].add(features["candidate_id"])
    
    # Convert to aggregated features
    aggregated = {}
    for req_id, data in req_data.items():
        aggregated[req_id] = {
            "req_id": req_id,
            "total_applications": data["internal_count"] + data["external_count"],
            "unique_candidates": len(data["candidate_ids"]),
            "internal_ratio": (
                data["internal_count"] / (data["internal_count"] + data["external_count"])
                if (data["internal_count"] + data["external_count"]) > 0 else 0
            ),
            "dominant_seniority": _get_mode(data["seniority_levels"]),
            "dominant_profession": _get_mode(data["profession_groups"]),
            "avg_quality_score": round(mean(data["quality_scores"]), 4) if data["quality_scores"] else None,
            "avg_skill_count": round(mean(data["skill_counts"]), 2) if data["skill_counts"] else None,
        }
    
    return aggregated


def _get_mode(items: List[str]) -> Optional[str]:
    """Get the most common item in a list."""
    if not items:
        return None
    from collections import Counter
    counter = Counter(items)
    return counter.most_common(1)[0][0]


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example usage with sample data using proper field names
    sample_doc = {
        "_id": "app_12345",
        Application.REQ_ID: "REQ_001",
        Application.CANDIDATE_ID: "CAND_789",
        Application.ATS_APPLICATION: {
            AtsApplication.INTERNAL_CANDIDATE_INFO: {"is_internal_candidate": False}
        },
        Application.ATS_SAMURAI: {
            SamuraiJson.EXPERIENCE: [
                {
                    SamuraiExperienceItem.SENIORITY: "senior",
                    "top_professions": [{"name": "IT_&_Software_Development", "score": 0.92}]
                }
            ]
        },
        "samurai_json": {
            SamuraiJson.SKILLS_KB: {
                SkillsKB.SKILLS_KB_ITEMS: [
                    {SkillsKBItem.SKILL_ENTITY: "Python", SkillsKBItem.SKILL_SCORE: 0.85},
                    {SkillsKBItem.SKILL_ENTITY: "Machine Learning", SkillsKBItem.SKILL_SCORE: 0.78},
                    {SkillsKBItem.SKILL_ENTITY: "SQL", SkillsKBItem.SKILL_SCORE: 0.92}
                ]
            }
        }
    }
    
    features = extract_static_features(sample_doc)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
