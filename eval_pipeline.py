#!/usr/bin/env python
"""
eval_pipeline.py - Main entrypoint for due sourcing model evaluation.

Evaluates a regression model for "additional candidates needed" prediction against
two baselines (segment-median and stage heuristic), computing:
- Regression accuracy (MAE, RMSE, sMAPE)
- Decision quality for "Need sourcing" binary classification
- Business-friendly asymmetric regret (under/over sourcing)
- Prioritization lift (top-decile analysis)

Usage:
    python eval_pipeline.py \
        --data_path data.csv \
        --preds_path preds.csv \
        --out_dir outputs \
        --time_cutoff 2025-10-01 \
        --snapshot_strategy day_7 \
        --need_k 5 \
        --c_under 3.0 \
        --c_over 1.0 \
        --segment_keys job_family seniority location is_remote \
        --heuristic_qualified_min 3 \
        --heuristic_interview_min 1 \
        --heuristic_recommend_n 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from io_utils import (
    load_data,
    validate_and_prepare,
    join_predictions,
    select_snapshots,
    time_split,
    ensure_output_dir,
)
from baselines import fit_baselines, predict_baselines
from metrics import compute_all_metrics, compute_segment_metrics, flatten_metrics
from report_generator import generate_html_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate due sourcing regression model against baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to main data file (CSV or Parquet)"
    )
    parser.add_argument(
        "--preds_path", type=str, default=None,
        help="Path to external predictions file (optional, if y_pred_model not in main data)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="outputs",
        help="Output directory for results"
    )
    
    # Split and snapshot settings
    parser.add_argument(
        "--time_cutoff", type=str, default="2025-10-01",
        help="Cutoff date for train/test split (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--snapshot_strategy", type=str, default="day_7",
        choices=["day_7", "first", "latest_before_fill", "all"],
        help="Strategy for selecting one snapshot per req_id"
    )
    
    # Metric parameters
    parser.add_argument(
        "--need_k", type=int, default=5,
        help="Threshold for 'need sourcing' decision (y >= need_k means need)"
    )
    parser.add_argument(
        "--c_under", type=float, default=3.0,
        help="Cost weight for under-sourcing (missing candidates)"
    )
    parser.add_argument(
        "--c_over", type=float, default=1.0,
        help="Cost weight for over-sourcing (wasted effort)"
    )
    
    # Segmentation
    parser.add_argument(
        "--segment_keys", nargs="+", default=["job_family", "seniority", "location", "is_remote"],
        help="Columns to use for segment-based analysis and baseline"
    )
    
    # Heuristic baseline parameters
    parser.add_argument(
        "--heuristic_qualified_min", type=int, default=3,
        help="Minimum qualified candidates for heuristic baseline"
    )
    parser.add_argument(
        "--heuristic_interview_min", type=int, default=1,
        help="Minimum interview candidates for heuristic baseline"
    )
    parser.add_argument(
        "--heuristic_recommend_n", type=int, default=5,
        help="Candidates to recommend if sourcing needed (heuristic baseline)"
    )
    
    # Options
    parser.add_argument(
        "--clamp_predictions", action="store_true", default=True,
        help="Clamp predictions to >= 0"
    )
    parser.add_argument(
        "--generate_plots", action="store_true", default=False,
        help="Generate diagnostic plots (requires matplotlib)"
    )
    
    return parser.parse_args()


def sanity_checks(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, need_k: int) -> Dict:
    """
    Run sanity checks on the data and print diagnostics.
    
    Args:
        df: Full dataset
        train_df: Training split
        test_df: Test split
        need_k: Threshold for need sourcing
        
    Returns:
        Dict with sanity check results
    """
    logger.info("=" * 60)
    logger.info("SANITY CHECKS")
    logger.info("=" * 60)
    
    # y_true distribution
    y_true = df["y_true"]
    logger.info(f"y_true: min={y_true.min():.0f}, median={y_true.median():.1f}, "
                f"max={y_true.max():.0f}, mean={y_true.mean():.2f}")
    
    # Label need distribution
    label_need = (y_true >= need_k).mean() * 100
    logger.info(f"% label_need (y_true >= {need_k}): {label_need:.1f}%")
    
    # Split sizes
    logger.info(f"Train: {len(train_df)} rows, {train_df['req_id'].nunique()} reqs")
    logger.info(f"Test: {len(test_df)} rows, {test_df['req_id'].nunique()} reqs")
    
    # Check overlap
    train_reqs = set(train_df["req_id"].unique())
    test_reqs = set(test_df["req_id"].unique())
    overlap = train_reqs & test_reqs
    logger.info(f"req_id overlap between splits: {len(overlap)} (should be 0)")
    
    if overlap:
        logger.error(f"OVERLAP DETECTED: {list(overlap)[:5]}...")
    
    logger.info("=" * 60)
    
    return {
        "y_true_min": float(y_true.min()),
        "y_true_median": float(y_true.median()),
        "y_true_max": float(y_true.max()),
        "y_true_mean": float(y_true.mean()),
        "pct_label_need": label_need,
        "train_rows": len(train_df),
        "train_reqs": train_df["req_id"].nunique(),
        "test_rows": len(test_df),
        "test_reqs": test_df["req_id"].nunique(),
        "overlap_count": len(overlap)
    }


def generate_plots(test_df: pd.DataFrame, out_dir: Path):
    """Generate diagnostic plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots")
        return
    
    # Histogram of y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(test_df["y_true"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Additional Candidates Needed (y_true)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of y_true in Test Set")
    fig.savefig(out_dir / "hist_y_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved histogram: hist_y_true.png")
    
    # Scatter: y_true vs y_pred_model (if available)
    if "y_pred_model" in test_df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(test_df["y_true"], test_df["y_pred_model"], alpha=0.5, s=20)
        max_val = max(test_df["y_true"].max(), test_df["y_pred_model"].max())
        ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred_model")
        ax.set_title("Model Predictions vs Ground Truth")
        ax.legend()
        fig.savefig(out_dir / "scatter_model.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved scatter plot: scatter_model.png")


def generate_report(
    config: Dict,
    sanity: Dict,
    overall_metrics: Dict[str, Dict],
    out_dir: Path
) -> str:
    """
    Generate a markdown report summarizing the evaluation.
    
    Args:
        config: Configuration dict
        sanity: Sanity check results
        overall_metrics: Dict mapping predictor names to their metrics
        out_dir: Output directory
        
    Returns:
        Markdown report string
    """
    lines = [
        "# Due Sourcing Model Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Data path:** `{config.get('data_path', 'N/A')}`",
        f"- **Time cutoff:** {config.get('time_cutoff', 'N/A')}",
        f"- **Snapshot strategy:** {config.get('snapshot_strategy', 'N/A')}",
        f"- **Need threshold (k):** {config.get('need_k', 'N/A')}",
        f"- **Under-sourcing cost (c_under):** {config.get('c_under', 'N/A')}",
        f"- **Over-sourcing cost (c_over):** {config.get('c_over', 'N/A')}",
        f"- **Segment keys:** {config.get('segment_keys', 'N/A')}",
        "",
        "## Dataset Summary",
        "",
        f"- **Train rows:** {sanity.get('train_rows', 'N/A')} ({sanity.get('train_reqs', 'N/A')} unique reqs)",
        f"- **Test rows:** {sanity.get('test_rows', 'N/A')} ({sanity.get('test_reqs', 'N/A')} unique reqs)",
        f"- **y_true stats:** min={sanity.get('y_true_min', 0):.0f}, "
        f"median={sanity.get('y_true_median', 0):.1f}, max={sanity.get('y_true_max', 0):.0f}",
        f"- **% needing sourcing (y_true >= k):** {sanity.get('pct_label_need', 0):.1f}%",
        "",
        "## Baseline Definitions",
        "",
        "### Segment-Median Baseline",
        "- Computes median(eventual_total_applicants) per segment on training data",
        "- Predicts: `y_pred = max(0, segment_median - current_applicants)`",
        "- Uses hierarchical fallback for unseen segments",
        "",
        "### Stage Heuristic Baseline",
        f"- Rule: Need sourcing if `qualified_count < {config.get('heuristic_qualified_min', 3)}` "
        f"OR `interview_count < {config.get('heuristic_interview_min', 1)}`",
        f"- If need: recommend {config.get('heuristic_recommend_n', 5)} candidates, else: 0",
        "",
        "## Overall Metrics Comparison",
        "",
    ]
    
    # Build comparison table
    predictors = list(overall_metrics.keys())
    if not predictors:
        lines.append("*No metrics computed*")
    else:
        # Table header
        headers = ["Metric"] + predictors
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Metric rows to include
        metric_keys = [
            ("MAE", lambda m: m.get("regression", {}).get("mae", "N/A")),
            ("RMSE", lambda m: m.get("regression", {}).get("rmse", "N/A")),
            ("sMAPE", lambda m: m.get("regression", {}).get("smape", "N/A")),
            (f"Precision@k={config.get('need_k', 5)}", lambda m: m.get("decision", {}).get("precision", "N/A")),
            (f"Recall@k={config.get('need_k', 5)}", lambda m: m.get("decision", {}).get("recall", "N/A")),
            ("F1", lambda m: m.get("decision", {}).get("f1", "N/A")),
            ("Mean Regret", lambda m: m.get("regret", {}).get("mean_regret", "N/A")),
            ("Lift Top10%", lambda m: m.get("lift", {}).get("lift_top10", "N/A")),
        ]
        
        for metric_name, getter in metric_keys:
            row = [metric_name]
            for pred in predictors:
                val = getter(overall_metrics.get(pred, {}))
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            lines.append("| " + " | ".join(row) + " |")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **MAE/RMSE**: Lower is better - measures prediction accuracy",
        "- **Precision**: Of reqs flagged as needing sourcing, what % actually needed it?",
        "- **Recall**: Of reqs that needed sourcing, what % did we correctly flag?",
        "- **Mean Regret**: Weighted cost of under/over-sourcing decisions",
        "- **Lift Top10%**: How much higher is avg y_true in top 10% predicted vs overall?",
        "",
        "## Output Files",
        "",
        f"- `{out_dir}/summary.json` - Complete metrics in JSON format",
        f"- `{out_dir}/metrics_overall.csv` - Overall metrics comparison",
        f"- `{out_dir}/metrics_by_segment_*.csv` - Segment-level breakdowns",
        f"- `{out_dir}/report.md` - This report",
        "",
    ])
    
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> Dict:
    """
    Main pipeline execution.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dict with all results
    """
    out_dir = ensure_output_dir(args.out_dir)
    
    # 1. Load and validate data
    logger.info("Step 1: Loading and validating data...")
    df = load_data(args.data_path)
    df = validate_and_prepare(df, args.segment_keys)
    
    # 2. Join external predictions if provided
    if args.preds_path:
        logger.info("Joining external predictions...")
        df = join_predictions(df, args.preds_path)
    
    # Check if model predictions exist
    has_model = "y_pred_model" in df.columns
    if not has_model:
        logger.warning("y_pred_model not found in data. Only baselines will be evaluated.")
    
    # 3. Snapshot selection
    logger.info(f"Step 2: Selecting snapshots (strategy={args.snapshot_strategy})...")
    df = select_snapshots(df, args.snapshot_strategy)
    
    # 4. Train/test split
    logger.info(f"Step 3: Splitting data (cutoff={args.time_cutoff})...")
    train_df, test_df = time_split(df, args.time_cutoff)
    
    # Sanity checks
    sanity = sanity_checks(df, train_df, test_df, args.need_k)
    
    # Handle empty splits
    if len(train_df) == 0:
        logger.error("Training set is empty! Check time_cutoff.")
        return {"error": "Empty training set"}
    if len(test_df) == 0:
        logger.error("Test set is empty! Check time_cutoff.")
        return {"error": "Empty test set"}
    
    # 5. Fit baselines on TRAIN
    logger.info("Step 4: Fitting baselines on training data...")
    seg_baseline, heur_baseline = fit_baselines(
        train_df,
        segment_keys=args.segment_keys,
        heuristic_qualified_min=args.heuristic_qualified_min,
        heuristic_interview_min=args.heuristic_interview_min,
        heuristic_recommend_n=args.heuristic_recommend_n
    )
    
    # 6. Generate baseline predictions on TEST
    logger.info("Step 5: Generating baseline predictions on test data...")
    test_df = predict_baselines(test_df, seg_baseline, heur_baseline)
    
    # 7. Compute metrics
    logger.info("Step 6: Computing metrics...")
    overall_metrics = {}
    
    # Model metrics (if available)
    if has_model:
        model_metrics = compute_all_metrics(
            test_df["y_true"].values,
            test_df["y_pred_model"].values,
            need_k=args.need_k,
            c_under=args.c_under,
            c_over=args.c_over,
            clamp_predictions=args.clamp_predictions
        )
        overall_metrics["model"] = model_metrics
        logger.info(f"Model MAE: {model_metrics['regression']['mae']:.4f}")
    
    # Segment baseline metrics
    seg_metrics = compute_all_metrics(
        test_df["y_true"].values,
        test_df["y_pred_baseline_seg"].values,
        need_k=args.need_k,
        c_under=args.c_under,
        c_over=args.c_over,
        clamp_predictions=args.clamp_predictions
    )
    overall_metrics["baseline_seg"] = seg_metrics
    logger.info(f"Segment baseline MAE: {seg_metrics['regression']['mae']:.4f}")
    
    # Heuristic baseline metrics
    heur_metrics = compute_all_metrics(
        test_df["y_true"].values,
        test_df["y_pred_baseline_heur"].values,
        need_k=args.need_k,
        c_under=args.c_under,
        c_over=args.c_over,
        clamp_predictions=args.clamp_predictions
    )
    overall_metrics["baseline_heur"] = heur_metrics
    logger.info(f"Heuristic baseline MAE: {heur_metrics['regression']['mae']:.4f}")
    
    # 8. Segment-level reporting
    logger.info("Step 7: Computing segment-level metrics...")
    available_seg_keys = [k for k in args.segment_keys if k in test_df.columns]
    
    # Metrics for each segment key individually
    for seg_key in available_seg_keys:
        for pred_name, pred_col in [
            ("model", "y_pred_model"),
            ("baseline_seg", "y_pred_baseline_seg"),
            ("baseline_heur", "y_pred_baseline_heur")
        ]:
            if pred_col not in test_df.columns:
                continue
            seg_df = compute_segment_metrics(
                test_df, "y_true", pred_col, seg_key,
                need_k=args.need_k, c_under=args.c_under, c_over=args.c_over
            )
            seg_df["predictor"] = pred_name
            
            out_path = out_dir / f"metrics_by_{seg_key}_{pred_name}.csv"
            seg_df.to_csv(out_path, index=False)
            logger.info(f"Saved segment metrics: {out_path}")
    
    # Combined segment combo (if multiple keys)
    if len(available_seg_keys) > 1:
        test_df["_segment_combo"] = test_df[available_seg_keys].astype(str).agg("|".join, axis=1)
        for pred_name, pred_col in [
            ("model", "y_pred_model"),
            ("baseline_seg", "y_pred_baseline_seg"),
            ("baseline_heur", "y_pred_baseline_heur")
        ]:
            if pred_col not in test_df.columns:
                continue
            seg_df = compute_segment_metrics(
                test_df, "y_true", pred_col, "_segment_combo",
                need_k=args.need_k, c_under=args.c_under, c_over=args.c_over
            )
            seg_df["predictor"] = pred_name
            
            out_path = out_dir / f"metrics_by_segment_combo_{pred_name}.csv"
            seg_df.to_csv(out_path, index=False)
    
    # 9. Save outputs
    logger.info("Step 8: Saving outputs...")
    
    # Config
    config = {
        "data_path": args.data_path,
        "preds_path": args.preds_path,
        "time_cutoff": args.time_cutoff,
        "snapshot_strategy": args.snapshot_strategy,
        "need_k": args.need_k,
        "c_under": args.c_under,
        "c_over": args.c_over,
        "segment_keys": args.segment_keys,
        "heuristic_qualified_min": args.heuristic_qualified_min,
        "heuristic_interview_min": args.heuristic_interview_min,
        "heuristic_recommend_n": args.heuristic_recommend_n,
    }
    
    # Summary JSON
    summary = {
        "config": config,
        "sanity": sanity,
        "overall": overall_metrics,
        "baseline_rules": {
            "segment_median": "median(eventual_total_applicants) per segment -> y_pred = max(0, median - current)",
            "heuristic": heur_baseline.get_rule_description()
        }
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary: {out_dir}/summary.json")
    
    # Overall metrics CSV
    rows = []
    for pred_name, metrics in overall_metrics.items():
        flat = flatten_metrics(metrics)
        flat["predictor"] = pred_name
        rows.append(flat)
    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)
    logger.info(f"Saved overall metrics: {out_dir}/metrics_overall.csv")
    
    # Train segment counts (for credibility)
    seg_summary = seg_baseline.get_train_segment_summary()
    seg_summary.to_csv(out_dir / "train_segment_counts.csv", index=False)
    logger.info(f"Saved train segment counts: {out_dir}/train_segment_counts.csv")
    
    # Markdown report
    report = generate_report(config, sanity, overall_metrics, out_dir)
    with open(out_dir / "report.md", "w") as f:
        f.write(report)
    logger.info(f"Saved report: {out_dir}/report.md")
    
    # HTML report (presentation-ready)
    baseline_rules = {
        "segment_median": "median(eventual_total_applicants) per segment -> y_pred = max(0, median - current)",
        "heuristic": heur_baseline.get_rule_description()
    }
    html_report = generate_html_report(
        config=config,
        sanity=sanity,
        overall_metrics=overall_metrics,
        out_dir=out_dir,
        baseline_rules=baseline_rules
    )
    with open(out_dir / "report.html", "w") as f:
        f.write(html_report)
    logger.info(f"Saved HTML report: {out_dir}/report.html")
    
    # 10. Optional plots
    if args.generate_plots:
        logger.info("Generating plots...")
        generate_plots(test_df, out_dir)
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {out_dir}")
    logger.info("=" * 60)
    
    return summary


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        result = run_pipeline(args)
        if "error" in result:
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

