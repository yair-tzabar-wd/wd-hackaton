"""
baselines.py - Baseline implementations for due sourcing prediction.

Implements:
1. Segment-median baseline: Predicts based on historical median eventual_total_applicants
   per segment, with hierarchical fallback for missing segments.
   
2. Stage heuristic baseline: Simple rule-based prediction based on current funnel stage counts.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SegmentMedianBaseline:
    """
    Segment-median baseline for predicting additional candidates needed.
    
    Prediction Rule:
    1. Compute segment_median_total = median(eventual_total_applicants) per segment on TRAIN
    2. For each test row: predicted_total = segment_median_total for that segment
    3. y_pred = max(0, predicted_total - current_applicants)
    
    Hierarchical Fallback:
    If a segment combination is not found in training:
    - Try progressively dropping segment keys from the end
    - e.g., job_family+seniority+location -> job_family+seniority -> job_family -> global
    - Always end with global median if nothing matches
    """
    
    def __init__(self, segment_keys: List[str]):
        """
        Args:
            segment_keys: List of column names to use for segmentation
        """
        self.segment_keys = segment_keys
        self.segment_medians: Dict[Tuple, float] = {}
        self.global_median: float = 0.0
        self.fallback_chain: List[List[str]] = []
        self.train_segment_counts: Dict[Tuple, int] = {}
        
    def fit(self, train_df: pd.DataFrame) -> "SegmentMedianBaseline":
        """
        Fit the baseline on training data.
        
        Computes median(eventual_total_applicants) for each segment combination
        and all fallback levels.
        
        Args:
            train_df: Training DataFrame with segment_keys and eventual_total_applicants
            
        Returns:
            self
        """
        # Filter to available segment keys
        available_keys = [k for k in self.segment_keys if k in train_df.columns]
        if not available_keys:
            logger.warning("No segment keys found in training data; using global median only")
        self.segment_keys = available_keys
        
        # Build fallback chain: full -> progressively drop last key -> global
        self.fallback_chain = []
        for i in range(len(self.segment_keys), 0, -1):
            self.fallback_chain.append(self.segment_keys[:i])
        
        # Compute global median (always available as final fallback)
        self.global_median = train_df["eventual_total_applicants"].median()
        logger.info(f"Global median eventual_total_applicants: {self.global_median:.2f}")
        
        # Compute medians for each level in fallback chain
        for keys in self.fallback_chain:
            if not keys:
                continue
            grouped = train_df.groupby(keys, dropna=False)["eventual_total_applicants"]
            medians = grouped.median()
            counts = grouped.size()
            
            for idx, median_val in medians.items():
                # Convert to tuple for dict key
                key = idx if isinstance(idx, tuple) else (idx,)
                self.segment_medians[key] = median_val
                self.train_segment_counts[key] = counts[idx]
        
        logger.info(f"Fitted segment medians for {len(self.segment_medians)} segment combinations")
        return self
    
    def _get_segment_key(self, row: pd.Series, keys: List[str]) -> Tuple:
        """Extract segment key tuple from a row."""
        return tuple(row.get(k, None) for k in keys)
    
    def _predict_total(self, row: pd.Series) -> float:
        """
        Predict eventual_total_applicants for a single row with fallback.
        
        Args:
            row: A row from the test DataFrame
            
        Returns:
            Predicted eventual_total_applicants
        """
        # Try each level in the fallback chain
        for keys in self.fallback_chain:
            if not keys:
                continue
            segment_key = self._get_segment_key(row, keys)
            if segment_key in self.segment_medians:
                return self.segment_medians[segment_key]
        
        # Final fallback: global median
        return self.global_median
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for test data.
        
        Args:
            test_df: Test DataFrame with segment_keys and current_applicants
            
        Returns:
            Array of y_pred = max(0, predicted_total - current_applicants)
        """
        predictions = []
        fallback_counts = {"full": 0, "partial": 0, "global": 0}
        
        for _, row in test_df.iterrows():
            predicted_total = self._predict_total(row)
            y_pred = max(0, predicted_total - row["current_applicants"])
            predictions.append(y_pred)
            
            # Track fallback usage for diagnostics
            full_key = self._get_segment_key(row, self.segment_keys)
            if full_key in self.segment_medians:
                fallback_counts["full"] += 1
            elif any(self._get_segment_key(row, keys) in self.segment_medians 
                     for keys in self.fallback_chain[1:]):
                fallback_counts["partial"] += 1
            else:
                fallback_counts["global"] += 1
        
        logger.info(f"Segment baseline predictions: {len(predictions)} total, "
                    f"full match={fallback_counts['full']}, "
                    f"partial fallback={fallback_counts['partial']}, "
                    f"global fallback={fallback_counts['global']}")
        
        return np.array(predictions)
    
    def get_train_segment_summary(self) -> pd.DataFrame:
        """Return summary of training segment counts and medians."""
        rows = []
        for key, median in self.segment_medians.items():
            count = self.train_segment_counts.get(key, 0)
            rows.append({
                "segment": str(key),
                "median_total": median,
                "train_count": count
            })
        return pd.DataFrame(rows)


class StageHeuristicBaseline:
    """
    Stage heuristic baseline for "need sourcing" prediction.
    
    A simple, interpretable rule-based baseline that uses funnel stage counts
    to decide if sourcing is needed.
    
    Default Heuristic (if stage columns exist):
        need = (qualified_count < qualified_min) OR (interview_count < interview_min)
        if need: y_pred = recommend_n
        else: y_pred = 0
    
    Fallback Heuristic (if stage columns missing):
        need = (current_applicants < qualified_min)
        if need: y_pred = recommend_n
        else: y_pred = 0
    
    This baseline is fast and interpretable, serving as a "business rule" comparison.
    """
    
    def __init__(
        self,
        qualified_min: int = 3,
        interview_min: int = 1,
        recommend_n: int = 5
    ):
        """
        Args:
            qualified_min: Minimum qualified candidates to not need sourcing
            interview_min: Minimum interview candidates to not need sourcing
            recommend_n: Number of candidates to recommend if sourcing needed
        """
        self.qualified_min = qualified_min
        self.interview_min = interview_min
        self.recommend_n = recommend_n
        self.has_stage_cols = False
        self.stage_cols_used: List[str] = []
        
    def fit(self, train_df: pd.DataFrame) -> "StageHeuristicBaseline":
        """
        Fit the heuristic (just checks which columns are available).
        
        The heuristic is rule-based and doesn't actually learn from training data,
        but we check column availability during fit.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            self
        """
        # Check for stage columns
        stage_cols = {
            "qualified_count": "qualified_count",
            "interview_count": "interview_count"
        }
        
        self.stage_cols_used = []
        for name, col in stage_cols.items():
            if col in train_df.columns:
                self.stage_cols_used.append(col)
        
        self.has_stage_cols = len(self.stage_cols_used) > 0
        
        if self.has_stage_cols:
            logger.info(f"Stage heuristic using columns: {self.stage_cols_used}")
        else:
            logger.info("Stage columns not found; heuristic will use current_applicants only")
        
        return self
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the heuristic rules.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Array of y_pred values
        """
        predictions = []
        need_sourcing_count = 0
        
        for _, row in test_df.iterrows():
            if self.has_stage_cols:
                # Use stage columns for decision
                qualified = row.get("qualified_count", 0) or 0
                interview = row.get("interview_count", 0) or 0
                
                need = (qualified < self.qualified_min) or (interview < self.interview_min)
            else:
                # Fallback: use current_applicants
                need = row["current_applicants"] < self.qualified_min
            
            if need:
                y_pred = self.recommend_n
                need_sourcing_count += 1
            else:
                y_pred = 0
            
            predictions.append(y_pred)
        
        pct_need = 100 * need_sourcing_count / len(predictions) if predictions else 0
        logger.info(f"Heuristic baseline: {need_sourcing_count}/{len(predictions)} "
                    f"({pct_need:.1f}%) flagged as needing sourcing")
        
        return np.array(predictions)
    
    def get_rule_description(self) -> str:
        """Return human-readable description of the heuristic rule."""
        if self.has_stage_cols:
            return (
                f"Need sourcing IF (qualified_count < {self.qualified_min}) "
                f"OR (interview_count < {self.interview_min})\n"
                f"IF need: recommend {self.recommend_n} candidates, ELSE: 0"
            )
        else:
            return (
                f"Need sourcing IF (current_applicants < {self.qualified_min})\n"
                f"IF need: recommend {self.recommend_n} candidates, ELSE: 0"
            )


def fit_baselines(
    train_df: pd.DataFrame,
    segment_keys: List[str],
    heuristic_qualified_min: int = 3,
    heuristic_interview_min: int = 1,
    heuristic_recommend_n: int = 5
) -> Tuple[SegmentMedianBaseline, StageHeuristicBaseline]:
    """
    Convenience function to fit both baselines.
    
    Args:
        train_df: Training DataFrame
        segment_keys: Segment columns for median baseline
        heuristic_qualified_min: Min qualified for heuristic
        heuristic_interview_min: Min interview for heuristic
        heuristic_recommend_n: Recommend N for heuristic
        
    Returns:
        Tuple of (segment_baseline, heuristic_baseline)
    """
    logger.info("Fitting segment-median baseline...")
    seg_baseline = SegmentMedianBaseline(segment_keys).fit(train_df)
    
    logger.info("Fitting stage heuristic baseline...")
    heur_baseline = StageHeuristicBaseline(
        qualified_min=heuristic_qualified_min,
        interview_min=heuristic_interview_min,
        recommend_n=heuristic_recommend_n
    ).fit(train_df)
    
    return seg_baseline, heur_baseline


def predict_baselines(
    test_df: pd.DataFrame,
    seg_baseline: SegmentMedianBaseline,
    heur_baseline: StageHeuristicBaseline
) -> pd.DataFrame:
    """
    Generate predictions from both baselines.
    
    Args:
        test_df: Test DataFrame
        seg_baseline: Fitted segment-median baseline
        heur_baseline: Fitted stage heuristic baseline
        
    Returns:
        DataFrame with added prediction columns
    """
    test_df = test_df.copy()
    test_df["y_pred_baseline_seg"] = seg_baseline.predict(test_df)
    test_df["y_pred_baseline_heur"] = heur_baseline.predict(test_df)
    return test_df

