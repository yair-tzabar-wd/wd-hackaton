"""
io_utils.py - Data loading, validation, joining predictions, and snapshot selection.

Handles:
- Loading CSV/Parquet files
- Validating required columns and types
- Computing y_true if missing
- Joining external predictions
- Snapshot selection strategies
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required columns for the pipeline
REQUIRED_COLS = ["req_id", "snapshot_date", "current_applicants", "eventual_total_applicants"]
OPTIONAL_STAGE_COLS = ["qualified_count", "interview_count", "onsite_count", "offer_count"]
DEFAULT_SEGMENT_KEYS = ["job_family", "seniority", "location", "is_remote"]

# Column name mappings for alternative dataset formats
COLUMN_MAPPINGS = {
    # target -> source alternatives
    "snapshot_date": ["target_date", "date", "snap_date"],
    "current_applicants": ["total_active", "active_count", "current_count"],
    "eventual_total_applicants": ["target_total_applicants", "final_applicants", "total_applicants"],
    "seniority": ["req_seniority", "level"],
    "job_family": ["req_top_profession", "profession", "job_category", "function"],
    "is_remote": ["req_is_remote", "remote"],
    "location": ["req_location", "region", "country"],
    "qualified_count": ["Screening_all", "screening_count"],
    "interview_count": ["Interview_all", "interviews"],
    "offer_count": ["Offer_all", "offers"],
}


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Parquet file.
    
    Args:
        data_path: Path to the data file (csv or parquet)
        
    Returns:
        DataFrame with loaded data
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .parquet")
    
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df


def apply_column_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column name mappings to handle alternative dataset formats.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    for target_col, source_alternatives in COLUMN_MAPPINGS.items():
        if target_col not in df.columns:
            for source_col in source_alternatives:
                if source_col in df.columns:
                    df[target_col] = df[source_col]
                    logger.info(f"Mapped column '{source_col}' -> '{target_col}'")
                    break
    
    return df


def compute_req_created_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute req_created_date from snapshot_date and days_since_open if available.
    
    Args:
        df: DataFrame with snapshot_date and optionally days_since_open
        
    Returns:
        DataFrame with req_created_date column
    """
    df = df.copy()
    
    if "req_created_date" not in df.columns:
        if "days_since_open" in df.columns:
            # req_created_date = snapshot_date - days_since_open
            logger.info("Computing req_created_date from snapshot_date and days_since_open")
            df["req_created_date"] = df["snapshot_date"] - pd.to_timedelta(df["days_since_open"], unit="D")
        else:
            # Fallback: use min snapshot_date per req_id
            logger.warning("req_created_date and days_since_open not found; using min snapshot_date per req_id")
            df["req_created_date"] = df.groupby("req_id")["snapshot_date"].transform("min")
    
    return df


def validate_and_prepare(df: pd.DataFrame, segment_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate required columns exist, parse types, compute y_true if missing.
    
    Args:
        df: Input DataFrame
        segment_keys: List of segment column names to validate
        
    Returns:
        Validated and prepared DataFrame
    """
    df = df.copy()
    
    # Apply column mappings for alternative dataset formats
    df = apply_column_mappings(df)
    
    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    # Parse dates
    for date_col in ["snapshot_date", "req_created_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Compute req_created_date if missing
    df = compute_req_created_date(df)
    
    # Ensure req_id is string
    df["req_id"] = df["req_id"].astype(str)
    
    # Ensure numeric columns
    for col in ["current_applicants", "eventual_total_applicants"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Compute y_true if missing
    if "y_true" not in df.columns:
        logger.info("y_true not found; computing as max(0, eventual_total_applicants - current_applicants)")
        df["y_true"] = np.maximum(0, df["eventual_total_applicants"] - df["current_applicants"])
    
    # Handle NaNs in critical columns
    n_before = len(df)
    df = df.dropna(subset=["y_true", "current_applicants", "snapshot_date", "req_created_date"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with NaN in critical columns")
    
    # Validate segment keys if provided
    if segment_keys:
        missing_seg = [k for k in segment_keys if k not in df.columns]
        if missing_seg:
            logger.warning(f"Segment keys not found in data: {missing_seg}. They will be ignored.")
    
    # Check for stage columns
    available_stage = [c for c in OPTIONAL_STAGE_COLS if c in df.columns]
    if available_stage:
        logger.info(f"Found stage columns: {available_stage}")
        for col in available_stage:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    else:
        logger.info("No stage columns found; heuristic baseline will use current_applicants only")
    
    return df


def join_predictions(
    df: pd.DataFrame, 
    preds_path: str,
    pred_col: str = "y_pred_model"
) -> pd.DataFrame:
    """
    Join external predictions by (req_id, snapshot_date).
    If snapshot_date granularity differs, match on req_id + nearest snapshot within same day.
    
    Args:
        df: Main DataFrame
        preds_path: Path to predictions file
        pred_col: Name of prediction column in preds file
        
    Returns:
        DataFrame with predictions joined
    """
    preds = load_data(preds_path)
    
    if pred_col not in preds.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in {preds_path}")
    
    preds["req_id"] = preds["req_id"].astype(str)
    if "snapshot_date" in preds.columns:
        preds["snapshot_date"] = pd.to_datetime(preds["snapshot_date"], errors="coerce")
    
    # Try exact join first
    df_merged = df.merge(
        preds[["req_id", "snapshot_date", pred_col]],
        on=["req_id", "snapshot_date"],
        how="left",
        suffixes=("", "_pred")
    )
    
    n_matched = df_merged[pred_col].notna().sum()
    logger.info(f"Exact join matched {n_matched}/{len(df)} rows")
    
    # If many unmatched, try fuzzy join by day
    n_unmatched = len(df) - n_matched
    if n_unmatched > 0.1 * len(df):
        logger.info("Attempting fuzzy join by date (same day, nearest snapshot)")
        
        # Create date columns for fuzzy matching
        df["_snap_date"] = df["snapshot_date"].dt.date
        preds["_snap_date"] = preds["snapshot_date"].dt.date
        
        # For unmatched, find nearest prediction on same day
        unmatched_mask = df_merged[pred_col].isna()
        unmatched_df = df[unmatched_mask].copy()
        
        fuzzy_matches = []
        for _, row in unmatched_df.iterrows():
            candidates = preds[
                (preds["req_id"] == row["req_id"]) & 
                (preds["_snap_date"] == row["_snap_date"])
            ]
            if len(candidates) > 0:
                # Find nearest by time
                time_diffs = abs(candidates["snapshot_date"] - row["snapshot_date"])
                nearest_idx = time_diffs.idxmin()
                fuzzy_matches.append({
                    "req_id": row["req_id"],
                    "snapshot_date": row["snapshot_date"],
                    pred_col: candidates.loc[nearest_idx, pred_col]
                })
        
        if fuzzy_matches:
            fuzzy_df = pd.DataFrame(fuzzy_matches)
            # Update unmatched rows
            for _, frow in fuzzy_df.iterrows():
                mask = (df_merged["req_id"] == frow["req_id"]) & \
                       (df_merged["snapshot_date"] == frow["snapshot_date"])
                df_merged.loc[mask, pred_col] = frow[pred_col]
            
            logger.info(f"Fuzzy join matched additional {len(fuzzy_matches)} rows")
        
        df.drop(columns=["_snap_date"], inplace=True, errors="ignore")
    
    n_final = df_merged[pred_col].notna().sum()
    n_missing = len(df_merged) - n_final
    if n_missing > 0:
        logger.warning(f"{n_missing} rows have no prediction after join")
    
    return df_merged


def select_snapshots(
    df: pd.DataFrame,
    strategy: str = "day_7"
) -> pd.DataFrame:
    """
    Select one snapshot per req_id to avoid correlated daily rows.
    
    Strategies:
    - "day_7": Pick snapshot closest to req_created_date + 7 days (>= day 7 preferred)
    - "first": Earliest snapshot per req_id
    - "latest_before_fill": Latest snapshot per req_id
    - "all": Keep all snapshots (with warning)
    
    Args:
        df: DataFrame with all snapshots
        strategy: Selection strategy name
        
    Returns:
        DataFrame with selected snapshots
    """
    valid_strategies = ["day_7", "first", "latest_before_fill", "all"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid snapshot_strategy: {strategy}. Choose from {valid_strategies}")
    
    n_before = len(df)
    n_reqs = df["req_id"].nunique()
    
    if strategy == "all":
        logger.warning(
            "Using 'all' snapshots - correlation between daily rows may inflate confidence intervals. "
            "Consider using 'day_7' for more independent samples."
        )
        return df
    
    if strategy == "day_7":
        # Target: req_created_date + 7 days
        df = df.copy()
        df["_target_date"] = df["req_created_date"] + pd.Timedelta(days=7)
        df["_days_from_target"] = (df["snapshot_date"] - df["_target_date"]).dt.total_seconds() / 86400
        
        # Prefer >= day 7, then nearest overall
        df["_pref_order"] = df["_days_from_target"].apply(
            lambda x: (0, abs(x)) if x >= 0 else (1, abs(x))  # (priority, distance)
        )
        
        # Sort by preference and pick first per req_id
        df = df.sort_values(["req_id", "_pref_order"])
        df = df.groupby("req_id", as_index=False).first()
        df = df.drop(columns=["_target_date", "_days_from_target", "_pref_order"], errors="ignore")
        
    elif strategy == "first":
        df = df.sort_values(["req_id", "snapshot_date"])
        df = df.groupby("req_id", as_index=False).first()
        
    elif strategy == "latest_before_fill":
        df = df.sort_values(["req_id", "snapshot_date"], ascending=[True, False])
        df = df.groupby("req_id", as_index=False).first()
    
    logger.info(f"Snapshot selection '{strategy}': {n_before} -> {len(df)} rows ({n_reqs} reqs)")
    return df


def time_split(
    df: pd.DataFrame,
    time_cutoff: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by req_created_date for time-based train/test split.
    Ensures no req_id overlap between train and test.
    
    Args:
        df: DataFrame to split
        time_cutoff: Cutoff date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    cutoff = pd.to_datetime(time_cutoff)
    
    train = df[df["req_created_date"] < cutoff].copy()
    test = df[df["req_created_date"] >= cutoff].copy()
    
    # Verify no overlap
    train_reqs = set(train["req_id"].unique())
    test_reqs = set(test["req_id"].unique())
    overlap = train_reqs & test_reqs
    
    if overlap:
        raise AssertionError(
            f"req_id overlap between train and test: {len(overlap)} reqs. "
            "This should not happen with time-based split on req_created_date."
        )
    
    logger.info(f"Time split at {time_cutoff}: train={len(train)} rows ({len(train_reqs)} reqs), "
                f"test={len(test)} rows ({len(test_reqs)} reqs)")
    
    return train, test


def ensure_output_dir(out_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

