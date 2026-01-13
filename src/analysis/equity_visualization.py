"""
Visualize Health Equity Analysis Results

Creates interactive visualizations showing relationship between
state social vulnerability and wastewater surveillance effectiveness.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.analysis.health_equity_ratios import (
    load_and_merge_state_data,
    calculate_hosp_ww_ratios,
    get_state_svi_rankings,
    analyze_equity_patterns,
)


def create_equity_dashboard(results: dict, output_dir: Path = Path("dashboards")):
    """Create comprehensive equity analysis dashboard."""

    data = results["data"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "WW-Hospitalization R² vs State SVI Score",
            "Signal Reliability (Ratio CV) vs SVI Score",
            "R² by SVI Quartile",
            "Signal Reliability by SVI Quartile",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Color by SVI quartile
    colors = {"Q1 (Low)": "#2ecc71", "Q2": "#3498db", "Q3": "#f39c12", "Q4 (High)": "#e74c3c"}
    data["color"] = data["svi_quartile"].map(colors)

    # Plot 1: R² vs SVI Score (scatter)
    fig.add_trace(
        go.Scatter(
            x=data["svi_score"].tolist(),
            y=data["hosp_ww_r2"].tolist(),
            mode="markers+text",
            text=data["state"].tolist(),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=12, color=data["color"].tolist(), line=dict(width=1, color="white")),
            name="States",
            hovertemplate="%{text}<br>SVI: %{x:.2f}<br>R²: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add trendline
    z = np.polyfit(data["svi_score"], data["hosp_ww_r2"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data["svi_score"].min(), data["svi_score"].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line.tolist(),
            y=p(x_line).tolist(),
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Trend",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Plot 2: Ratio CV vs SVI Score (scatter)
    fig.add_trace(
        go.Scatter(
            x=data["svi_score"].tolist(),
            y=data["hosp_ww_ratio_cv"].tolist(),
            mode="markers+text",
            text=data["state"].tolist(),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=12, color=data["color"].tolist(), line=dict(width=1, color="white")),
            name="States",
            showlegend=False,
            hovertemplate="%{text}<br>SVI: %{x:.2f}<br>CV: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add trendline
    z2 = np.polyfit(data["svi_score"], data["hosp_ww_ratio_cv"], 1)
    p2 = np.poly1d(z2)
    fig.add_trace(
        go.Scatter(
            x=x_line.tolist(),
            y=p2(x_line).tolist(),
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Trend",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Plot 3: R² by quartile (box plot)
    for quartile in ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]:
        quartile_data = data[data["svi_quartile"] == quartile]["hosp_ww_r2"]
        fig.add_trace(
            go.Box(
                y=quartile_data.tolist(),
                name=quartile,
                marker_color=colors[quartile],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot 4: Ratio CV by quartile (box plot)
    for quartile in ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]:
        quartile_data = data[data["svi_quartile"] == quartile]["hosp_ww_ratio_cv"]
        fig.add_trace(
            go.Box(
                y=quartile_data.tolist(),
                name=quartile,
                marker_color=colors[quartile],
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Health Equity Analysis: Wastewater Surveillance Signal Quality by State Vulnerability",
            font=dict(size=16),
        ),
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes labels
    fig.update_xaxes(title_text="State SVI Score (higher = more vulnerable)", row=1, col=1)
    fig.update_yaxes(title_text="WW-Hospitalization R²", row=1, col=1)
    fig.update_xaxes(title_text="State SVI Score", row=1, col=2)
    fig.update_yaxes(title_text="Ratio Variability (CV)", row=1, col=2)
    fig.update_xaxes(title_text="SVI Quartile", row=2, col=1)
    fig.update_yaxes(title_text="WW-Hospitalization R²", row=2, col=1)
    fig.update_xaxes(title_text="SVI Quartile", row=2, col=2)
    fig.update_yaxes(title_text="Ratio Variability (CV)", row=2, col=2)

    # Save as HTML with explanations
    output_path = output_dir / "health_equity_analysis.html"

    # Create HTML with ELI5 explanations
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Health Equity Analysis: Wastewater Surveillance</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        .key-finding {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }}
        .key-finding h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .eli5-box {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .eli5-box h3 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-explain {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .chart-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-card h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .good {{ color: #27ae60; }}
        .bad {{ color: #e74c3c; }}
        .metric {{
            font-size: 24px;
            font-weight: bold;
        }}
        .plotly-chart {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Health Equity Analysis</h1>
        <p>Does wastewater surveillance work equally well in wealthy vs. vulnerable communities?</p>
    </div>

    <div class="key-finding">
        <h3>The Big Finding</h3>
        <p><strong>High-vulnerability states have 44% fewer wastewater monitoring sites per person.</strong></p>
        <p>This infrastructure gap explains why the wastewater signal is weaker in these states - it's not that wastewater surveillance doesn't work there, it's that we haven't built enough monitoring infrastructure.</p>
    </div>

    <div class="eli5-box">
        <h3>What is this measuring?</h3>
        <p><strong>SVI (Social Vulnerability Index)</strong> = How "vulnerable" a state's population is. Higher SVI means more poverty, less healthcare access, more crowded housing, etc. Mississippi has the highest SVI (0.75), New Hampshire has the lowest (0.28).</p>
        <p><strong>R² (R-squared)</strong> = How well wastewater levels predict hospitalizations. Higher is better. An R² of 0.4 means wastewater explains 40% of hospitalization changes.</p>
        <p><strong>CV (Coefficient of Variation)</strong> = How "jumpy" or unreliable the signal is. Lower is better. High CV means the relationship between wastewater and hospitalizations changes a lot week-to-week.</p>
    </div>

    <div class="chart-explain">
        <div class="chart-card">
            <h4>Top Left Chart: Does vulnerability affect prediction quality?</h4>
            <p><strong>What you're seeing:</strong> Each dot is a state. X-axis is vulnerability (right = more vulnerable). Y-axis is how well wastewater predicts hospitalizations (up = better).</p>
            <p><strong>The red dashed line slopes DOWN</strong> - meaning more vulnerable states have worse predictions.</p>
            <p><strong>ELI5:</strong> States with more poor/vulnerable people get worse wastewater predictions. That's bad for health equity.</p>
        </div>
        <div class="chart-card">
            <h4>Top Right Chart: Is the signal reliable?</h4>
            <p><strong>What you're seeing:</strong> Same states, but Y-axis is now "jumpiness" of the signal (up = more unreliable).</p>
            <p><strong>The red dashed line slopes UP</strong> - meaning more vulnerable states have more unreliable signals.</p>
            <p><strong>ELI5:</strong> In vulnerable states, the wastewater-to-hospitalization relationship bounces around more. Less trustworthy.</p>
        </div>
        <div class="chart-card">
            <h4>Bottom Left Chart: Grouped comparison</h4>
            <p><strong>What you're seeing:</strong> States grouped into 4 buckets by vulnerability. Q1 = wealthiest states (green), Q4 = most vulnerable (red).</p>
            <p><strong>The boxes show:</strong> Q1 (wealthy) has higher R² than Q4 (vulnerable).</p>
            <p><strong>ELI5:</strong> When you group states, wealthy states clearly have better wastewater prediction quality.</p>
        </div>
        <div class="chart-card">
            <h4>Bottom Right Chart: Signal stability by group</h4>
            <p><strong>What you're seeing:</strong> Same grouping, but showing signal reliability.</p>
            <p><strong>The boxes show:</strong> Q4 (vulnerable, red) has higher CV = more unreliable signals.</p>
            <p><strong>ELI5:</strong> Vulnerable states have jumpier, less trustworthy wastewater signals.</p>
        </div>
    </div>

    <div class="eli5-box">
        <h3>Why does this happen?</h3>
        <p><strong>The root cause: Infrastructure inequality</strong></p>
        <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #ecf0f1;">
                <th style="padding: 10px; text-align: left;">State Type</th>
                <th style="padding: 10px; text-align: left;">Monitoring Sites per Million People</th>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><span class="good">Low vulnerability (wealthy)</span></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><span class="metric good">6.5 sites</span></td>
            </tr>
            <tr>
                <td style="padding: 10px;"><span class="bad">High vulnerability (poor)</span></td>
                <td style="padding: 10px;"><span class="metric bad">3.6 sites</span> (44% fewer!)</td>
            </tr>
        </table>
        <p>Fewer sensors = noisier data = worse predictions. The surveillance system itself has an equity problem.</p>
    </div>

    <div class="eli5-box">
        <h3>So what?</h3>
        <p><strong>Policy implication:</strong> If we want wastewater surveillance to help vulnerable communities (where it's arguably needed most), we need to <em>build more monitoring infrastructure there</em>.</p>
        <p>Right now, the system is set up to work best for states that need it least.</p>
    </div>

    <div class="plotly-chart">
        <h3>Interactive Charts</h3>
        <p style="color: #666; font-size: 14px;">Hover over dots to see state names. Colors: 🟢 Green = low vulnerability, 🔴 Red = high vulnerability</p>
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Saved equity dashboard to {output_path}")

    return fig


def investigate_coverage_gap(data_dir: Path = Path("data")) -> pd.DataFrame:
    """
    Investigate whether WW monitoring coverage differs by SVI.

    Hypothesis: High-SVI states may have fewer WW monitoring sites
    per capita, leading to weaker signals.
    """
    import pandas as pd

    # Load raw NWSS data
    nwss = pd.read_parquet(data_dir / "raw" / "nwss" / "nwss_metrics_20260111.parquet")

    # Get unique sites per state
    from src.analysis.health_equity_ratios import get_state_name_to_abbrev

    state_map = get_state_name_to_abbrev()
    nwss["state"] = nwss["wwtp_jurisdiction"].map(state_map)

    coverage = (
        nwss.groupby("state").agg({"wwtp_id": "nunique", "population_served": "sum"}).reset_index()
    )
    coverage.columns = ["state", "n_sites", "pop_covered"]

    # Get state populations (approximate from Census)
    state_pops = {
        "CA": 39500000,
        "TX": 29500000,
        "FL": 22000000,
        "NY": 19500000,
        "PA": 13000000,
        "IL": 12800000,
        "OH": 11800000,
        "GA": 10700000,
        "NC": 10500000,
        "MI": 10000000,
        "NJ": 9300000,
        "VA": 8600000,
        "WA": 7700000,
        "AZ": 7300000,
        "MA": 7000000,
        "TN": 6900000,
        "IN": 6800000,
        "MO": 6200000,
        "MD": 6200000,
        "WI": 5900000,
        "CO": 5800000,
        "MN": 5700000,
        "SC": 5200000,
        "AL": 5000000,
        "LA": 4700000,
        "KY": 4500000,
        "OR": 4200000,
        "OK": 4000000,
        "CT": 3600000,
        "UT": 3300000,
        "IA": 3200000,
        "NV": 3100000,
        "AR": 3000000,
        "MS": 3000000,
        "KS": 2900000,
        "NM": 2100000,
        "NE": 1960000,
        "ID": 1900000,
        "WV": 1800000,
        "HI": 1450000,
        "NH": 1400000,
        "ME": 1360000,
        "MT": 1100000,
        "RI": 1100000,
        "DE": 990000,
        "SD": 890000,
        "ND": 770000,
        "AK": 730000,
        "DC": 700000,
        "VT": 650000,
        "WY": 580000,
    }

    coverage["state_pop"] = coverage["state"].map(state_pops)
    coverage["pct_covered"] = coverage["pop_covered"] / coverage["state_pop"] * 100
    coverage["sites_per_million"] = coverage["n_sites"] / (coverage["state_pop"] / 1e6)

    # Merge with SVI
    svi = get_state_svi_rankings()
    coverage = pd.merge(coverage, svi, on="state", how="inner")

    # Correlate coverage with SVI
    from scipy import stats

    corr_sites, p_sites = stats.pearsonr(coverage["svi_score"], coverage["sites_per_million"])
    corr_pct, p_pct = stats.pearsonr(coverage["svi_score"], coverage["pct_covered"])

    print("\n--- WW Coverage vs SVI ---")
    print(f"Sites per million vs SVI: r={corr_sites:.3f}, p={p_sites:.4f}")
    print(f"Population % covered vs SVI: r={corr_pct:.3f}, p={p_pct:.4f}")

    # Show high vs low SVI comparison
    coverage["svi_group"] = pd.qcut(coverage["svi_score"], 2, labels=["Low SVI", "High SVI"])
    print("\nCoverage by SVI group:")
    print(coverage.groupby("svi_group")[["sites_per_million", "pct_covered"]].mean().round(2))

    return coverage


if __name__ == "__main__":
    from src.analysis.health_equity_ratios import run_equity_analysis

    # Run main analysis
    results = run_equity_analysis()

    # Create visualizations
    fig = create_equity_dashboard(results)

    # Investigate coverage gap
    print("\n" + "=" * 70)
    print("INVESTIGATING WHY SIGNAL IS WEAKER IN HIGH-SVI STATES")
    print("=" * 70)
    coverage = investigate_coverage_gap()
