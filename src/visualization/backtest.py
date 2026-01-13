"""
Backtesting visualization for the forecasting model.

Trains on historical data, makes predictions, and compares to actuals.
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_backtest(
    data_dir: Path = Path("data"),
    target: str = "respiratory_total",
    test_weeks: int = 20,
    horizon: int = 1,
    output_dir: Path = Path("dashboards")
) -> go.Figure:
    """
    Run walk-forward backtesting.

    For each week in the test period:
    1. Train on all data up to that point
    2. Predict the next `horizon` weeks
    3. Compare to actual values
    """
    from src.models.multi_pathogen import load_and_merge_data, create_multi_pathogen_features

    logger.info("Loading data...")
    merged = load_and_merge_data(data_dir)
    df = create_multi_pathogen_features(merged)

    # Aggregate to national level for cleaner visualization
    national = df.groupby('week_end_date').agg({
        'covid_hosp': 'sum',
        'flu_hosp': 'sum',
        'rsv_hosp': 'sum',
        'respiratory_total': 'sum',
        'covid_ww_percentile': 'mean',
        'flu_ww_conc': 'mean',
    }).reset_index()

    national['week_end_date'] = pd.to_datetime(national['week_end_date'])
    national = national.sort_values('week_end_date').reset_index(drop=True)

    # Create lag features for national data
    for lag in [1, 2, 3]:
        national[f'{target}_lag{lag}'] = national[target].shift(lag)
    national[f'{target}_roll2'] = national[target].rolling(2).mean()
    national['week_of_year'] = national['week_end_date'].dt.isocalendar().week.astype(int)
    national['month'] = national['week_end_date'].dt.month

    # Define features
    feature_cols = [f'{target}_lag1', f'{target}_lag2', f'{target}_lag3',
                    f'{target}_roll2', 'week_of_year', 'month',
                    'covid_ww_percentile']

    # Drop rows with NaN features
    national = national.dropna(subset=feature_cols + [target]).reset_index(drop=True)

    # Split into train period and test period
    n_total = len(national)
    test_start_idx = n_total - test_weeks

    logger.info(f"Total weeks: {n_total}, Test weeks: {test_weeks}")
    logger.info(f"Test period: {national.iloc[test_start_idx]['week_end_date']} to {national.iloc[-1]['week_end_date']}")

    # Walk-forward backtesting
    results = []

    for i in range(test_start_idx, n_total - horizon + 1):
        # Train on all data up to this point
        train_df = national.iloc[:i]

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict next week(s)
        for h in range(horizon):
            pred_idx = i + h
            if pred_idx >= n_total:
                break

            X_pred = national.iloc[pred_idx:pred_idx+1][feature_cols].values
            prediction = model.predict(X_pred)[0]
            actual = national.iloc[pred_idx][target]
            pred_date = national.iloc[pred_idx]['week_end_date']

            results.append({
                'date': pred_date,
                'actual': actual,
                'predicted': prediction,
                'horizon': h + 1,
                'train_size': len(train_df)
            })

    results_df = pd.DataFrame(results)

    # Calculate metrics
    mae = np.mean(np.abs(results_df['actual'] - results_df['predicted']))
    mape = np.mean(np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual'])) * 100

    logger.info(f"Backtest MAE: {mae:.0f}")
    logger.info(f"Backtest MAPE: {mape:.1f}%")

    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Backtest: {target.replace("_", " ").title()} - Actual vs Predicted',
            'Prediction Error Over Time'
        ),
        row_heights=[0.7, 0.3]
    )

    # Convert to lists for Plotly
    dates = results_df['date'].dt.strftime('%Y-%m-%d').tolist()
    actuals = results_df['actual'].tolist()
    predictions = results_df['predicted'].tolist()
    errors = (results_df['predicted'] - results_df['actual']).tolist()
    pct_errors = ((results_df['predicted'] - results_df['actual']) / results_df['actual'] * 100).tolist()

    # Actual values
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actuals,
            name='Actual',
            line=dict(color='#1d3557', width=3)
        ),
        row=1, col=1
    )

    # Predicted values
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predictions,
            name='Predicted',
            line=dict(color='#e63946', width=2, dash='dash')
        ),
        row=1, col=1
    )

    # Error bars
    fig.add_trace(
        go.Bar(
            x=dates,
            y=pct_errors,
            name='% Error',
            marker_color=['#2a9d8f' if e >= 0 else '#e63946' for e in pct_errors]
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=f'Model Backtest Results (MAE: {mae:.0f}, MAPE: {mape:.1f}%)',
        height=700,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Hospitalizations", row=1, col=1)
    fig.update_yaxes(title_text="Error %", row=2, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=1)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "backtest_results.html"
    fig.write_html(output_path)
    logger.info(f"Saved backtest visualization to {output_path}")

    # Also create a summary table
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Target: {target}")
    print(f"Test period: {test_weeks} weeks")
    print(f"Forecast horizon: {horizon} week(s)")
    print(f"\nMetrics:")
    print(f"  MAE:  {mae:,.0f} admissions")
    print(f"  MAPE: {mape:.1f}%")
    print(f"\nInterpretation:")
    if mape < 10:
        print("  Excellent - predictions within 10% of actual")
    elif mape < 20:
        print("  Good - predictions within 20% of actual")
    elif mape < 30:
        print("  Fair - predictions within 30% of actual")
    else:
        print("  Poor - predictions often off by >30%")

    return fig


if __name__ == "__main__":
    fig = run_backtest()

    import subprocess
    subprocess.run(["open", "dashboards/backtest_results.html"])
