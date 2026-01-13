"""
Visualization Dashboard for Respiratory Disease Forecasting

Generates interactive HTML dashboards showing:
1. Historical trends with wastewater signals
2. Forecast visualizations with confidence intervals
3. Model performance metrics
4. State-level comparisons
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_time_series_plot(
    df: pd.DataFrame, state: str = "US", output_path: Path | None = None
) -> go.Figure:
    """
    Create interactive time series plot showing hospitalizations and wastewater signals.
    """
    if state != "US":
        df = df[df["state"] == state].copy()
    else:
        # Aggregate to national level
        df = (
            df.groupby("week_end_date")
            .agg(
                {
                    "covid_hosp": "sum",
                    "flu_hosp": "sum",
                    "rsv_hosp": "sum",
                    "covid_ww_percentile": "mean",
                    "flu_ww_conc": "mean",
                }
            )
            .reset_index()
        )

    # Ensure datetime type and sort
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    df = df.sort_values("week_end_date").reset_index(drop=True)

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Hospital Admissions", "Wastewater Signals"),
        row_heights=[0.6, 0.4],
    )

    # Convert to native Python types for proper JSON serialization
    dates = df["week_end_date"].dt.strftime("%Y-%m-%d").tolist()

    # Hospital admissions
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["covid_hosp"].tolist(),
            name="COVID-19",
            line=dict(color="#e63946", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["flu_hosp"].tolist(),
            name="Influenza",
            line=dict(color="#457b9d", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=df["rsv_hosp"].tolist(), name="RSV", line=dict(color="#2a9d8f", width=2)
        ),
        row=1,
        col=1,
    )

    # Wastewater signals
    if "covid_ww_percentile" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["covid_ww_percentile"].tolist(),
                name="COVID Wastewater %ile",
                line=dict(color="#e63946", width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    if "flu_ww_conc" in df.columns:
        # Normalize flu concentration for visualization
        flu_norm = (df["flu_ww_conc"] / df["flu_ww_conc"].max() * 100).tolist()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=flu_norm,
                name="Flu Wastewater (norm)",
                line=dict(color="#457b9d", width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"Respiratory Disease Trends - {state}",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="Admissions", row=1, col=1)
    fig.update_yaxes(title_text="Percentile", row=2, col=1)

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved time series plot to {output_path}")

    return fig


def create_forecast_plot(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str = "respiratory_total",
    state: str = "US",
    output_path: Path | None = None,
    backtest_weeks: int = None,  # None = entire dataset
) -> go.Figure:
    """
    Create forecast visualization with confidence intervals and backtested predictions.
    """
    import xgboost as xgb

    # Filter data
    if state != "US":
        hist = historical_df[historical_df["state"] == state].copy()
        fcast = forecast_df[
            (forecast_df["state"] == state) & (forecast_df["target"] == target)
        ].copy()
    else:
        hist = historical_df.groupby("week_end_date").agg({target: "sum"}).reset_index()
        fcast = (
            forecast_df[forecast_df["target"] == target]
            .groupby("target_date")
            .agg({"point_estimate": "sum", "lower_bound": "sum", "upper_bound": "sum"})
            .reset_index()
        )

    # Ensure datetime types and sort
    hist["week_end_date"] = pd.to_datetime(hist["week_end_date"])
    hist = hist.sort_values("week_end_date").reset_index(drop=True)
    fcast["target_date"] = pd.to_datetime(fcast["target_date"])
    fcast = fcast.sort_values("target_date").reset_index(drop=True)

    # Generate backtested predictions
    backtest_results = _generate_backtest_predictions(hist, target, backtest_weeks)

    fig = go.Figure()

    # Convert to native Python types
    hist_dates = hist["week_end_date"].dt.strftime("%Y-%m-%d").tolist()
    hist_values = hist[target].tolist()

    # Historical data (actual)
    fig.add_trace(
        go.Scatter(x=hist_dates, y=hist_values, name="Actual", line=dict(color="#1d3557", width=2))
    )

    # Backtested predictions
    if len(backtest_results) > 0:
        bt_dates = backtest_results["date"].dt.strftime("%Y-%m-%d").tolist()
        bt_preds = backtest_results["predicted"].tolist()

        fig.add_trace(
            go.Scatter(
                x=bt_dates,
                y=bt_preds,
                name="Backtest (1-week)",
                line=dict(color="#2a9d8f", width=2, dash="dot"),
                opacity=0.8,
            )
        )

    # Future forecast with confidence interval
    if len(fcast) > 0:
        fcast_dates = fcast["target_date"].dt.strftime("%Y-%m-%d").tolist()
        fcast_upper = fcast["upper_bound"].tolist()
        fcast_lower = fcast["lower_bound"].tolist()
        fcast_point = fcast["point_estimate"].tolist()

        # Add confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                x=fcast_dates + fcast_dates[::-1],
                y=fcast_upper + fcast_lower[::-1],
                fill="toself",
                fillcolor="rgba(230, 57, 70, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

        # Point estimate
        fig.add_trace(
            go.Scatter(
                x=fcast_dates,
                y=fcast_point,
                name="Forecast",
                line=dict(color="#e63946", width=3),
                mode="lines+markers",
                marker=dict(size=10),
            )
        )

    # Calculate backtest accuracy for title
    if len(backtest_results) > 0:
        mape = (
            np.mean(
                np.abs(
                    (backtest_results["actual"] - backtest_results["predicted"])
                    / backtest_results["actual"]
                )
            )
            * 100
        )
        title = (
            f'{target.replace("_", " ").title()} Forecast - {state} (Backtest MAPE: {mape:.1f}%)'
        )
    else:
        title = f'{target.replace("_", " ").title()} Forecast - {state}'

    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Admissions",
        height=500,
        showlegend=True,
        hovermode="x unified",
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved forecast plot to {output_path}")

    return fig


def _generate_backtest_predictions(
    hist: pd.DataFrame, target: str, backtest_weeks: int = None
) -> pd.DataFrame:
    """
    Generate backtested 1-week predictions for the historical data.
    If backtest_weeks is None, backtest the entire dataset (minus training warmup).
    """
    import xgboost as xgb

    min_train_weeks = 10  # Minimum weeks needed for training

    if backtest_weeks is None:
        # Backtest entire dataset minus warmup period
        backtest_weeks = len(hist) - min_train_weeks

    if len(hist) < min_train_weeks + 5:
        return pd.DataFrame()

    # Create features
    df = hist.copy()
    for lag in [1, 2, 3]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df[f"{target}_roll2"] = df[target].rolling(2).mean()
    df["pct_change"] = df[target].pct_change(1) * 100
    df["week_of_year"] = df["week_end_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_end_date"].dt.month

    feature_cols = [
        f"{target}_lag1",
        f"{target}_lag2",
        f"{target}_lag3",
        f"{target}_roll2",
        "pct_change",
        "week_of_year",
        "month",
    ]

    # Clean data
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Walk-forward backtest
    test_start_idx = len(df) - backtest_weeks
    results = []

    for i in range(test_start_idx, len(df)):
        train_df = df.iloc[:i]
        if len(train_df) < 10:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

        X_pred = df.iloc[i : i + 1][feature_cols].values
        xgb_pred = model.predict(X_pred)[0]

        # Hybrid approach: blend with trend
        recent_pct_change = train_df["pct_change"].iloc[-1] if len(train_df) > 0 else 0
        recent_value = train_df[target].iloc[-1]
        trend_pred = recent_value * (1 + recent_pct_change / 100 * 0.8)

        is_surge = abs(recent_pct_change) > 15
        blend_weight = 0.4 if is_surge else 0.1
        prediction = (1 - blend_weight) * xgb_pred + blend_weight * trend_pred

        results.append(
            {
                "date": df.iloc[i]["week_end_date"],
                "actual": df.iloc[i][target],
                "predicted": max(0, prediction),
            }
        )

    return pd.DataFrame(results)


def create_state_comparison(
    df: pd.DataFrame, target: str = "respiratory_total", output_path: Path | None = None
) -> go.Figure:
    """
    Create state-level comparison heatmap.
    """
    # Pivot to state x week
    pivot = df.pivot_table(
        index="state", columns="week_end_date", values=target, aggfunc="sum"
    ).fillna(0)

    # Sort by total
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # Convert to native Python types
    z_data = pivot.values.tolist()
    x_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in pivot.columns]
    y_states = pivot.index.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data, x=x_dates, y=y_states, colorscale="Reds", colorbar=dict(title="Admissions")
        )
    )

    fig.update_layout(
        title=f'State Comparison - {target.replace("_", " ").title()}',
        xaxis_title="Week",
        yaxis_title="State",
        height=800,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved state comparison to {output_path}")

    return fig


def create_model_performance_plot(cv_results: dict, output_path: Path | None = None) -> go.Figure:
    """
    Create model performance comparison chart.
    """
    targets = list(cv_results.keys())
    mae_means = [cv_results[t]["cv_mae_mean"] for t in targets]
    mae_stds = [cv_results[t]["cv_mae_std"] for t in targets]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=targets,
            y=mae_means,
            error_y=dict(type="data", array=mae_stds),
            marker_color=["#e63946", "#457b9d", "#2a9d8f", "#1d3557"],
        )
    )

    fig.update_layout(
        title="Model Performance (Cross-Validation MAE)",
        xaxis_title="Target",
        yaxis_title="Mean Absolute Error",
        height=400,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved performance plot to {output_path}")

    return fig


def create_lead_time_analysis(df: pd.DataFrame, output_path: Path | None = None) -> go.Figure:
    """
    Analyze lead time correlation between wastewater and hospitalizations.
    """
    # Calculate correlations at different lags
    lags = range(0, 5)
    correlations = {"lag": [], "covid_corr": [], "flu_corr": []}

    for lag in lags:
        # Shift hospitalizations forward (wastewater leads)
        df_lagged = df.copy()
        df_lagged["covid_hosp_shifted"] = df_lagged.groupby("state")["covid_hosp"].shift(-lag)
        df_lagged["flu_hosp_shifted"] = df_lagged.groupby("state")["flu_hosp"].shift(-lag)

        covid_corr = df_lagged["covid_ww_percentile"].corr(df_lagged["covid_hosp_shifted"])
        flu_corr = df_lagged["flu_ww_conc"].corr(df_lagged["flu_hosp_shifted"])

        correlations["lag"].append(lag)
        correlations["covid_corr"].append(covid_corr)
        correlations["flu_corr"].append(flu_corr)

    corr_df = pd.DataFrame(correlations)

    # Convert to native Python types, replacing NaN with 0
    lag_values = corr_df["lag"].tolist()
    covid_corr_values = [v if pd.notna(v) else 0.0 for v in corr_df["covid_corr"]]
    flu_corr_values = [v if pd.notna(v) else 0.0 for v in corr_df["flu_corr"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lag_values,
            y=covid_corr_values,
            name="COVID-19",
            mode="lines+markers",
            line=dict(color="#e63946", width=3),
            marker=dict(size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lag_values,
            y=flu_corr_values,
            name="Influenza",
            mode="lines+markers",
            line=dict(color="#457b9d", width=3),
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title="Wastewater Lead Time Analysis",
        xaxis_title="Weeks Ahead (Wastewater Leading)",
        yaxis_title="Correlation",
        height=400,
        showlegend=True,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved lead time analysis to {output_path}")

    return fig


def create_full_dashboard(
    data_dir: Path = Path("data"),
    forecast_path: Path = Path("forecasts/latest_forecasts.csv"),
    output_dir: Path = Path("dashboards"),
) -> Path:
    """
    Generate complete HTML dashboard with all visualizations.
    """
    from src.models.multi_pathogen import load_and_merge_data, create_multi_pathogen_features

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    merged = load_and_merge_data(data_dir)
    featured = create_multi_pathogen_features(merged)

    # Load forecasts
    forecast_df = pd.read_csv(forecast_path, parse_dates=["target_date", "generated_at"])

    # Generate individual plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Creating time series plot...")
    ts_fig = create_time_series_plot(featured, state="US")

    logger.info("Creating forecast plot...")
    forecast_fig = create_forecast_plot(
        featured, forecast_df, target="respiratory_total", state="US"
    )

    logger.info("Creating state comparison...")
    state_fig = create_state_comparison(featured, target="respiratory_total")

    logger.info("Creating lead time analysis...")
    lead_fig = create_lead_time_analysis(featured)

    # Create combined dashboard
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Respiratory Disease Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1d3557, #457b9d);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #1d3557;
        }}
        .metric-card .change {{
            color: #e63946;
            font-size: 0.9em;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Respiratory Disease Forecasting Dashboard</h1>
        <p>Real-time predictions using wastewater surveillance data | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <h3>Total Forecast (2 weeks)</h3>
            <div class="value">{forecast_df['point_estimate'].sum():,.0f}</div>
            <div class="change">All respiratory admissions</div>
        </div>
        <div class="metric-card">
            <h3>COVID-19 Forecast</h3>
            <div class="value">{forecast_df[forecast_df['target']=='covid_hosp']['point_estimate'].sum():,.0f}</div>
            <div class="change">Hospitalizations</div>
        </div>
        <div class="metric-card">
            <h3>Influenza Forecast</h3>
            <div class="value">{forecast_df[forecast_df['target']=='flu_hosp']['point_estimate'].sum():,.0f}</div>
            <div class="change">Hospitalizations</div>
        </div>
        <div class="metric-card">
            <h3>RSV Forecast</h3>
            <div class="value">{forecast_df[forecast_df['target']=='rsv_hosp']['point_estimate'].sum():,.0f}</div>
            <div class="change">Hospitalizations</div>
        </div>
    </div>

    <div class="chart-container">
        <div id="timeseries"></div>
    </div>

    <div class="chart-container">
        <div id="forecast"></div>
    </div>

    <div class="grid-2">
        <div class="chart-container">
            <div id="leadtime"></div>
        </div>
        <div class="chart-container">
            <h3 style="margin-top:0; color:#1d3557;">Forecast Details</h3>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:10px; text-align:left;">Target</th>
                    <th style="padding:10px; text-align:right;">Forecast</th>
                    <th style="padding:10px; text-align:right;">95% CI</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td style="padding:10px; border-bottom:1px solid #eee;">{target}</td>
                    <td style="padding:10px; text-align:right; border-bottom:1px solid #eee;">{forecast_df[forecast_df['target']==target]['point_estimate'].sum():,.0f}</td>
                    <td style="padding:10px; text-align:right; border-bottom:1px solid #eee;">[{forecast_df[forecast_df['target']==target]['lower_bound'].sum():,.0f}, {forecast_df[forecast_df['target']==target]['upper_bound'].sum():,.0f}]</td>
                </tr>
                ''' for target in forecast_df['target'].unique()])}
            </table>
        </div>
    </div>

    <div class="chart-container">
        <div id="statemap"></div>
    </div>

    <div class="footer">
        <p>Data sources: CDC NWSS, NHSN | Model: XGBoost with wastewater signals</p>
    </div>

    <script>
        var ts_data = {ts_fig.to_json()};
        Plotly.newPlot('timeseries', ts_data.data, ts_data.layout);

        var forecast_data = {forecast_fig.to_json()};
        Plotly.newPlot('forecast', forecast_data.data, forecast_data.layout);

        var lead_data = {lead_fig.to_json()};
        Plotly.newPlot('leadtime', lead_data.data, lead_data.layout);

        var state_data = {state_fig.to_json()};
        Plotly.newPlot('statemap', state_data.data, state_data.layout);
    </script>
</body>
</html>
"""

    dashboard_path = output_dir / f"dashboard_{timestamp}.html"
    dashboard_path.write_text(dashboard_html)

    # Also save as latest
    latest_path = output_dir / "latest_dashboard.html"
    latest_path.write_text(dashboard_html)

    logger.info(f"Dashboard saved to {dashboard_path}")
    return dashboard_path


if __name__ == "__main__":
    create_full_dashboard()
