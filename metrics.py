"""
metrics.py - Metric functions for evaluation pipeline.

Implements:
- Regression metrics: MAE, RMSE, sMAPE
- Decision quality metrics: Precision, Recall, F1, Confusion matrix for "need sourcing"
- Business proxy: Asymmetric regret (under/over sourcing)
- Prioritization: Lift in top-decile predicted shortage
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Symmetric Mean Absolute Percentage Error (safe version).
    
    sMAPE = mean( 2 * |pred - true| / (|pred| + |true| + epsilon) )
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        sMAPE value in [0, 2]
    """
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return float(np.mean(numerator / denominator))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dict with MAE, RMSE, sMAPE
    """
    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred)
    }


def compute_decision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    need_k: int = 5
) -> Dict[str, float]:
    """
    Compute decision quality metrics for "Need sourcing" classification.
    
    Decision rule:
    - label_need = (y_true >= need_k)
    - pred_need = (y_pred >= need_k)
    
    Args:
        y_true: Ground truth additional candidates needed
        y_pred: Predicted additional candidates needed
        need_k: Threshold for "need sourcing" decision
        
    Returns:
        Dict with precision, recall, f1, and confusion matrix counts
    """
    label_need = (y_true >= need_k).astype(int)
    pred_need = (y_pred >= need_k).astype(int)
    
    # Handle edge cases where one class is missing
    if label_need.sum() == 0:
        logger.warning("No positive samples in y_true for need_k threshold")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif pred_need.sum() == 0:
        logger.warning("No positive predictions for need_k threshold")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precision_score(label_need, pred_need, zero_division=0)
        recall = recall_score(label_need, pred_need, zero_division=0)
        f1 = f1_score(label_need, pred_need, zero_division=0)
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(label_need, pred_need, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "need_k": need_k,
        "pct_label_need": float(label_need.mean() * 100),
        "pct_pred_need": float(pred_need.mean() * 100)
    }


def compute_regret(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_under: float = 3.0,
    c_over: float = 1.0
) -> Dict[str, float]:
    """
    Compute asymmetric business regret for under/over sourcing.
    
    regret = c_under * max(0, y_true - y_pred) + c_over * max(0, y_pred - y_true)
    
    Under-sourcing (y_true > y_pred): Miss candidates, bad for business
    Over-sourcing (y_pred > y_true): Waste sourcing effort, less critical
    
    Args:
        y_true: Ground truth additional candidates needed
        y_pred: Predicted additional candidates needed
        c_under: Cost weight for under-sourcing
        c_over: Cost weight for over-sourcing
        
    Returns:
        Dict with mean, median, and p90 regret
    """
    under = np.maximum(0, y_true - y_pred)
    over = np.maximum(0, y_pred - y_true)
    regret = c_under * under + c_over * over
    
    return {
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
        "c_under": c_under,
        "c_over": c_over,
        "total_under_cost": float(c_under * under.sum()),
        "total_over_cost": float(c_over * over.sum())
    }


def compute_lift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentiles: Tuple[int, ...] = (10, 20)
) -> Dict[str, float]:
    """
    Compute prioritization lift for top-decile (and optionally top-20%) predicted shortage.
    
    Lift measures how well the model identifies high-shortage reqs.
    
    Args:
        y_true: Ground truth additional candidates needed
        y_pred: Predicted additional candidates needed
        percentiles: Top percentiles to compute lift for
        
    Returns:
        Dict with avg_true for top-X%, overall avg, and lift ratios
    """
    if len(y_true) == 0:
        return {"avg_true_overall": 0.0}
    
    # Sort by prediction descending
    sort_idx = np.argsort(-y_pred)
    y_true_sorted = y_true[sort_idx]
    
    avg_overall = float(np.mean(y_true))
    results = {"avg_true_overall": avg_overall}
    
    for pct in percentiles:
        n_top = max(1, int(len(y_true) * pct / 100))
        top_true = y_true_sorted[:n_top]
        avg_top = float(np.mean(top_true))
        lift = avg_top / (avg_overall + 1e-9)
        
        results[f"avg_true_top{pct}"] = avg_top
        results[f"lift_top{pct}"] = lift
        results[f"n_top{pct}"] = n_top
    
    return results


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    need_k: int = 5,
    c_under: float = 3.0,
    c_over: float = 1.0,
    clamp_predictions: bool = True
) -> Dict[str, Dict]:
    """
    Compute all metrics for a single predictor.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        need_k: Threshold for "need sourcing" decision
        c_under: Cost weight for under-sourcing
        c_over: Cost weight for over-sourcing
        clamp_predictions: If True, clamp predictions to >= 0
        
    Returns:
        Nested dict with regression, decision, regret, and lift metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if clamp_predictions:
        y_pred = np.maximum(0, y_pred)
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"Dropping {n_invalid} rows with NaN in y_true or y_pred")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.error("No valid samples for metric computation")
        return {}
    
    return {
        "n_samples": len(y_true),
        "regression": compute_regression_metrics(y_true, y_pred),
        "decision": compute_decision_metrics(y_true, y_pred, need_k),
        "regret": compute_regret(y_true, y_pred, c_under, c_over),
        "lift": compute_lift(y_true, y_pred)
    }


def compute_segment_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    segment_col: str,
    need_k: int = 5,
    c_under: float = 3.0,
    c_over: float = 1.0
) -> pd.DataFrame:
    """
    Compute metrics grouped by a segment column.
    
    Args:
        df: DataFrame with predictions and truth
        y_true_col: Column name for ground truth
        y_pred_col: Column name for predictions
        segment_col: Column name to group by
        need_k: Threshold for "need sourcing"
        c_under: Under-sourcing cost weight
        c_over: Over-sourcing cost weight
        
    Returns:
        DataFrame with metrics per segment
    """
    results = []
    
    for segment_val, group in df.groupby(segment_col, dropna=False):
        y_true = group[y_true_col].values
        y_pred = group[y_pred_col].values
        
        # Skip if too few samples
        if len(y_true) < 2:
            continue
        
        # Basic metrics
        mae = compute_mae(y_true, y_pred)
        
        # Decision metrics
        label_need = (y_true >= need_k).astype(int)
        pred_need = (y_pred >= need_k).astype(int)
        
        if label_need.sum() > 0 and pred_need.sum() > 0:
            precision = precision_score(label_need, pred_need, zero_division=0)
            recall = recall_score(label_need, pred_need, zero_division=0)
        else:
            precision = 0.0
            recall = 0.0
        
        # Regret
        regret_dict = compute_regret(y_true, y_pred, c_under, c_over)
        
        results.append({
            "segment": str(segment_val),
            "count": len(group),
            "mae": mae,
            "precision": precision,
            "recall": recall,
            "mean_regret": regret_dict["mean_regret"],
            "pct_label_need": label_need.mean() * 100
        })
    
    return pd.DataFrame(results)


def flatten_metrics(metrics: Dict, prefix: str = "") -> Dict[str, float]:
    """Flatten nested metrics dict for easy CSV export."""
    flat = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, f"{key}_"))
        else:
            flat[key] = v
    return flat

