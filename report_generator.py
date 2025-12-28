"""
report_generator.py - Generate beautiful HTML reports for the evaluation pipeline.

Creates a visually appealing, presentation-ready HTML report with:
- Modern styling and typography
- Metric cards with visual indicators
- Comparison tables
- Inline SVG charts
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import html


def escape(text: Any) -> str:
    """Safely escape text for HTML."""
    return html.escape(str(text))


def format_metric(value: float, decimals: int = 2, is_percent: bool = False) -> str:
    """Format a metric value for display."""
    if is_percent:
        return f"{value:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def get_color_for_metric(metric_name: str, value: float, baseline_value: Optional[float] = None) -> str:
    """Get a color indicator based on metric performance."""
    # For metrics where lower is better
    lower_better = ["mae", "rmse", "smape", "mean_regret", "median_regret", "p90_regret"]
    # For metrics where higher is better
    higher_better = ["precision", "recall", "f1", "lift_top10", "lift_top20"]
    
    if baseline_value is not None:
        metric_lower = metric_name.lower()
        if any(m in metric_lower for m in lower_better):
            if value < baseline_value * 0.9:
                return "#10b981"  # Green - better
            elif value > baseline_value * 1.1:
                return "#ef4444"  # Red - worse
        elif any(m in metric_lower for m in higher_better):
            if value > baseline_value * 1.1:
                return "#10b981"  # Green - better
            elif value < baseline_value * 0.9:
                return "#ef4444"  # Red - worse
    
    return "#6366f1"  # Default indigo


def generate_bar_chart_svg(data: Dict[str, float], title: str, width: int = 400, height: int = 200) -> str:
    """Generate a simple horizontal bar chart as SVG."""
    if not data:
        return ""
    
    max_val = max(data.values()) if data.values() else 1
    bar_height = 30
    gap = 10
    label_width = 120
    chart_width = width - label_width - 20
    
    colors = ["#6366f1", "#8b5cf6", "#a855f7", "#d946ef"]
    
    svg_height = len(data) * (bar_height + gap) + 40
    
    svg = f'''<svg width="{width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
        <text x="{width/2}" y="20" text-anchor="middle" font-family="Inter, sans-serif" font-size="14" font-weight="600" fill="#1f2937">{escape(title)}</text>
    '''
    
    y = 40
    for i, (label, value) in enumerate(data.items()):
        bar_width = (value / max_val) * chart_width if max_val > 0 else 0
        color = colors[i % len(colors)]
        
        svg += f'''
        <text x="{label_width - 10}" y="{y + bar_height/2 + 5}" text-anchor="end" font-family="Inter, sans-serif" font-size="12" fill="#4b5563">{escape(label)}</text>
        <rect x="{label_width}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="{color}" opacity="0.8"/>
        <text x="{label_width + bar_width + 8}" y="{y + bar_height/2 + 5}" font-family="Inter, sans-serif" font-size="12" fill="#1f2937" font-weight="500">{value:.2f}</text>
        '''
        y += bar_height + gap
    
    svg += '</svg>'
    return svg


def generate_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊", color: str = "#6366f1") -> str:
    """Generate an HTML metric card."""
    return f'''
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div class="metric-icon">{icon}</div>
        <div class="metric-content">
            <div class="metric-title">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-subtitle">{escape(subtitle)}</div>
        </div>
    </div>
    '''


def generate_comparison_table(metrics: Dict[str, Dict], need_k: int = 5) -> str:
    """Generate HTML comparison table for predictors."""
    predictors = list(metrics.keys())
    if not predictors:
        return "<p>No metrics available</p>"
    
    rows = [
        ("MAE", "regression", "mae", "↓ Lower is better", False),
        ("RMSE", "regression", "rmse", "↓ Lower is better", False),
        ("sMAPE", "regression", "smape", "↓ Lower is better", False),
        (f"Precision @k={need_k}", "decision", "precision", "↑ Higher is better", True),
        (f"Recall @k={need_k}", "decision", "recall", "↑ Higher is better", True),
        ("F1 Score", "decision", "f1", "↑ Higher is better", False),
        ("Mean Regret", "regret", "mean_regret", "↓ Lower is better", False),
        ("P90 Regret", "regret", "p90_regret", "↓ Lower is better", False),
        ("Lift Top 10%", "lift", "lift_top10", "↑ Higher is better", False),
    ]
    
    html_parts = ['''
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric</th>
    ''']
    
    for pred in predictors:
        display_name = pred.replace("_", " ").title()
        html_parts.append(f'<th>{escape(display_name)}</th>')
    
    html_parts.append('<th>Interpretation</th></tr></thead><tbody>')
    
    for metric_name, category, key, interpretation, is_percent in rows:
        html_parts.append(f'<tr><td class="metric-name">{escape(metric_name)}</td>')
        
        values = []
        for pred in predictors:
            val = metrics.get(pred, {}).get(category, {}).get(key, None)
            values.append(val)
        
        # Find best value for highlighting
        valid_values = [v for v in values if v is not None]
        if valid_values:
            if "↓" in interpretation:
                best_val = min(valid_values)
            else:
                best_val = max(valid_values)
        else:
            best_val = None
        
        for val in values:
            if val is not None:
                if is_percent:
                    formatted = f"{val*100:.1f}%"
                else:
                    formatted = f"{val:.4f}"
                
                is_best = best_val is not None and abs(val - best_val) < 0.0001
                css_class = "best-value" if is_best else ""
                html_parts.append(f'<td class="{css_class}">{formatted}</td>')
            else:
                html_parts.append('<td class="na-value">N/A</td>')
        
        html_parts.append(f'<td class="interpretation">{escape(interpretation)}</td></tr>')
    
    html_parts.append('</tbody></table>')
    return ''.join(html_parts)


def generate_confusion_matrix_html(metrics: Dict[str, Dict], predictor: str) -> str:
    """Generate confusion matrix visualization."""
    decision = metrics.get(predictor, {}).get("decision", {})
    tp = decision.get("tp", 0)
    fp = decision.get("fp", 0)
    tn = decision.get("tn", 0)
    fn = decision.get("fn", 0)
    
    total = tp + fp + tn + fn
    if total == 0:
        return ""
    
    return f'''
    <div class="confusion-matrix">
        <div class="cm-title">Confusion Matrix: {escape(predictor.replace("_", " ").title())}</div>
        <div class="cm-grid">
            <div class="cm-corner"></div>
            <div class="cm-header">Pred: No</div>
            <div class="cm-header">Pred: Yes</div>
            <div class="cm-label">Actual: No</div>
            <div class="cm-cell tn">TN<br><strong>{tn}</strong></div>
            <div class="cm-cell fp">FP<br><strong>{fp}</strong></div>
            <div class="cm-label">Actual: Yes</div>
            <div class="cm-cell fn">FN<br><strong>{fn}</strong></div>
            <div class="cm-cell tp">TP<br><strong>{tp}</strong></div>
        </div>
    </div>
    '''


def generate_html_report(
    config: Dict,
    sanity: Dict,
    overall_metrics: Dict[str, Dict],
    out_dir: Path,
    baseline_rules: Optional[Dict] = None
) -> str:
    """
    Generate a beautiful HTML report.
    
    Args:
        config: Pipeline configuration
        sanity: Sanity check results
        overall_metrics: Metrics for all predictors
        out_dir: Output directory
        baseline_rules: Optional baseline rule descriptions
        
    Returns:
        HTML report string
    """
    need_k = config.get("need_k", 5)
    predictors = list(overall_metrics.keys())
    
    # Build metric cards for key stats
    metric_cards = []
    
    # Dataset cards
    metric_cards.append(generate_metric_card(
        "Training Samples",
        str(sanity.get("train_rows", 0)),
        f"{sanity.get('train_reqs', 0)} unique requisitions",
        "📚",
        "#3b82f6"
    ))
    
    metric_cards.append(generate_metric_card(
        "Test Samples",
        str(sanity.get("test_rows", 0)),
        f"{sanity.get('test_reqs', 0)} unique requisitions",
        "🧪",
        "#8b5cf6"
    ))
    
    metric_cards.append(generate_metric_card(
        "Need Sourcing Rate",
        f"{sanity.get('pct_label_need', 0):.1f}%",
        f"y_true >= {need_k}",
        "🎯",
        "#f59e0b"
    ))
    
    metric_cards.append(generate_metric_card(
        "Y_True Range",
        f"{sanity.get('y_true_min', 0):.0f} - {sanity.get('y_true_max', 0):.0f}",
        f"Median: {sanity.get('y_true_median', 0):.1f}",
        "📈",
        "#10b981"
    ))
    
    # Generate MAE comparison chart
    mae_data = {}
    for pred in predictors:
        mae = overall_metrics.get(pred, {}).get("regression", {}).get("mae")
        if mae is not None:
            mae_data[pred.replace("_", " ").title()] = mae
    mae_chart = generate_bar_chart_svg(mae_data, "Mean Absolute Error (MAE)")
    
    # Generate Regret comparison chart
    regret_data = {}
    for pred in predictors:
        regret = overall_metrics.get(pred, {}).get("regret", {}).get("mean_regret")
        if regret is not None:
            regret_data[pred.replace("_", " ").title()] = regret
    regret_chart = generate_bar_chart_svg(regret_data, "Mean Business Regret")
    
    # Generate Lift comparison chart
    lift_data = {}
    for pred in predictors:
        lift = overall_metrics.get(pred, {}).get("lift", {}).get("lift_top10")
        if lift is not None:
            lift_data[pred.replace("_", " ").title()] = lift
    lift_chart = generate_bar_chart_svg(lift_data, "Prioritization Lift (Top 10%)")
    
    # Generate confusion matrices
    confusion_matrices = ""
    for pred in predictors:
        confusion_matrices += generate_confusion_matrix_html(overall_metrics, pred)
    
    # Baseline rules section
    baseline_section = ""
    if baseline_rules:
        baseline_section = '''
        <section class="section">
            <h2>📋 Baseline Definitions</h2>
            <div class="baseline-rules">
        '''
        for name, rule in baseline_rules.items():
            baseline_section += f'''
            <div class="baseline-rule">
                <h4>{escape(name.replace("_", " ").title())}</h4>
                <pre>{escape(rule)}</pre>
            </div>
            '''
        baseline_section += '</div></section>'
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Due Sourcing Model Evaluation Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --border-color: #475569;
            --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --gradient-2: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: var(--gradient-1);
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            opacity: 0.5;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }}
        
        header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}
        
        header .timestamp {{
            font-size: 0.875rem;
            opacity: 0.7;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }}
        
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
        }}
        
        .section h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.25rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}
        
        .metric-icon {{
            font-size: 2rem;
            line-height: 1;
        }}
        
        .metric-content {{
            flex: 1;
        }}
        
        .metric-title {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }}
        
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .metric-subtitle {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}
        
        .chart-container {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .chart-container svg text {{
            fill: var(--text-primary) !important;
        }}
        
        .chart-container svg rect {{
            opacity: 0.9;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .metrics-table th,
        .metrics-table td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .metrics-table th {{
            background: var(--bg-card);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metrics-table td {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        
        .metrics-table tr:hover {{
            background: rgba(99, 102, 241, 0.1);
        }}
        
        .metric-name {{
            font-weight: 500;
            color: var(--text-primary);
        }}
        
        .best-value {{
            color: var(--accent-success) !important;
            font-weight: 600;
        }}
        
        .na-value {{
            color: var(--text-muted);
            font-style: italic;
        }}
        
        .interpretation {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        .confusion-matrices {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}
        
        .confusion-matrix {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
        }}
        
        .cm-title {{
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
            color: var(--text-secondary);
        }}
        
        .cm-grid {{
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            gap: 4px;
            font-size: 0.85rem;
        }}
        
        .cm-corner {{
            background: transparent;
        }}
        
        .cm-header {{
            background: var(--bg-secondary);
            padding: 0.75rem;
            text-align: center;
            font-weight: 500;
            color: var(--text-secondary);
            border-radius: 6px;
        }}
        
        .cm-label {{
            background: var(--bg-secondary);
            padding: 0.75rem;
            text-align: right;
            font-weight: 500;
            color: var(--text-secondary);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }}
        
        .cm-cell {{
            padding: 1rem;
            text-align: center;
            border-radius: 6px;
            font-weight: 500;
        }}
        
        .cm-cell strong {{
            display: block;
            font-size: 1.5rem;
            margin-top: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .cm-cell.tp {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-success);
        }}
        
        .cm-cell.tn {{
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
        }}
        
        .cm-cell.fp {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--accent-warning);
        }}
        
        .cm-cell.fn {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-danger);
        }}
        
        .baseline-rules {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        
        .baseline-rule {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.25rem;
        }}
        
        .baseline-rule h4 {{
            color: var(--accent-primary);
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }}
        
        .baseline-rule pre {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
        }}
        
        .config-item {{
            background: var(--bg-card);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .config-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        
        .config-value {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            color: var(--accent-primary);
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        
        .highlight {{
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            header h1 {{
                font-size: 1.75rem;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Due Sourcing Model Evaluation</h1>
            <p class="subtitle">Regression & Decision Quality Analysis</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </header>
        
        <section class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="metrics-grid">
                {''.join(metric_cards)}
            </div>
        </section>
        
        <section class="section">
            <h2>⚙️ Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <span class="config-label">Data Path</span>
                    <span class="config-value">{escape(config.get('data_path', 'N/A'))}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Time Cutoff</span>
                    <span class="config-value">{escape(config.get('time_cutoff', 'N/A'))}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Snapshot Strategy</span>
                    <span class="config-value">{escape(config.get('snapshot_strategy', 'N/A'))}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Need Threshold (k)</span>
                    <span class="config-value">{config.get('need_k', 5)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Under-sourcing Cost</span>
                    <span class="config-value">{config.get('c_under', 3.0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Over-sourcing Cost</span>
                    <span class="config-value">{config.get('c_over', 1.0)}</span>
                </div>
            </div>
        </section>
        
        {baseline_section}
        
        <section class="section">
            <h2>📈 Performance Comparison</h2>
            {generate_comparison_table(overall_metrics, need_k)}
        </section>
        
        <section class="section">
            <h2>📉 Visual Comparisons</h2>
            <div class="charts-grid">
                <div class="chart-container">{mae_chart}</div>
                <div class="chart-container">{regret_chart}</div>
                <div class="chart-container">{lift_chart}</div>
            </div>
        </section>
        
        <section class="section">
            <h2>🔢 Confusion Matrices</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                Decision threshold: y_pred ≥ {need_k} → "Need Sourcing"
            </p>
            <div class="confusion-matrices">
                {confusion_matrices}
            </div>
        </section>
        
        <section class="section">
            <h2>💡 Key Insights</h2>
            <div class="baseline-rules">
                <div class="baseline-rule">
                    <h4>What the Metrics Mean</h4>
                    <pre>MAE/RMSE: Prediction accuracy (lower = better)
Precision: Of flagged reqs, how many truly needed sourcing?
Recall: Of reqs needing sourcing, how many did we catch?
Regret: Business cost of wrong decisions
Lift: How well we prioritize high-need reqs</pre>
                </div>
                <div class="baseline-rule">
                    <h4>Business Impact</h4>
                    <pre>Under-sourcing cost: {config.get('c_under', 3.0)}x
Over-sourcing cost: {config.get('c_over', 1.0)}x

A model with high recall catches more true shortages.
A model with high precision reduces wasted effort.
Lower regret = better business outcomes.</pre>
                </div>
            </div>
        </section>
        
        <footer>
            <p>Generated by the Due Sourcing Evaluation Pipeline</p>
            <p style="margin-top: 0.5rem;">Built for HR Tech Hackathon 🚀</p>
        </footer>
    </div>
</body>
</html>
'''
    
    return html_content


def save_html_report(
    summary_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate HTML report from a summary.json file.
    
    Args:
        summary_path: Path to summary.json
        output_path: Optional output path for HTML file
        
    Returns:
        Path to saved HTML file
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    config = summary.get("config", {})
    sanity = summary.get("sanity", {})
    overall = summary.get("overall", {})
    baseline_rules = summary.get("baseline_rules", {})
    
    html_content = generate_html_report(
        config=config,
        sanity=sanity,
        overall_metrics=overall,
        out_dir=Path(summary_path).parent,
        baseline_rules=baseline_rules
    )
    
    if output_path is None:
        output_path = str(Path(summary_path).parent / "report.html")
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <summary.json> [output.html]")
        sys.exit(1)
    
    summary_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_path = save_html_report(summary_path, output_path)
    print(f"HTML report saved to: {result_path}")

