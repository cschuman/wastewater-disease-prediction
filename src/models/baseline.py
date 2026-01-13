"""
Baseline Models for Wastewater Disease Prediction

This module implements baseline models for predicting hospital admissions
from wastewater surveillance data:
1. Naive (last value)
2. Seasonal Naive (same week last year)
3. ARIMA with wastewater covariates
4. XGBoost with engineered features

Usage:
    python -m src.models.baseline
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    predictions: np.ndarray
    actuals: np.ndarray


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare the merged wastewater-hospital dataset."""
    df = pd.read_parquet(data_path)

    # Ensure proper date sorting
    date_col = 'week_end_date' if 'week_end_date' in df.columns else 'week_sat'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(['state', date_col])

    logger.info(f"Loaded {len(df):,} records from {data_path}")
    return df


def create_features(df: pd.DataFrame, target_col: str = 'covid') -> pd.DataFrame:
    """
    Create features for prediction models.

    Features include:
    - Lagged wastewater signals (1-4 weeks)
    - Lagged target values (1-4 weeks)
    - Rolling statistics
    - Time features (week of year, month)
    """
    df = df.copy()
    date_col = 'week_end_date' if 'week_end_date' in df.columns else 'week_sat'

    # Wastewater feature columns
    ww_cols = ['percentile', 'percentile_weighted', 'ptc_15d']
    ww_cols = [c for c in ww_cols if c in df.columns]

    # Create features per state
    feature_dfs = []

    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        state_df = state_df.sort_values(date_col)

        # Lagged wastewater features
        for col in ww_cols:
            for lag in [1, 2, 3, 4]:
                state_df[f'{col}_lag{lag}'] = state_df[col].shift(lag)

        # Lagged target
        for lag in [1, 2, 3, 4]:
            state_df[f'{target_col}_lag{lag}'] = state_df[target_col].shift(lag)

        # Rolling statistics on wastewater
        if 'percentile' in ww_cols:
            state_df['percentile_roll2_mean'] = state_df['percentile'].rolling(2).mean()
            state_df['percentile_roll4_mean'] = state_df['percentile'].rolling(4).mean()
            state_df['percentile_roll2_std'] = state_df['percentile'].rolling(2).std()

        # Rolling target statistics
        state_df[f'{target_col}_roll2_mean'] = state_df[target_col].rolling(2).mean()
        state_df[f'{target_col}_roll4_mean'] = state_df[target_col].rolling(4).mean()

        # Time features
        state_df['week_of_year'] = pd.to_datetime(state_df[date_col]).dt.isocalendar().week.astype(int)
        state_df['month'] = pd.to_datetime(state_df[date_col]).dt.month

        feature_dfs.append(state_df)

    result = pd.concat(feature_dfs, ignore_index=True)
    logger.info(f"Created {len(result.columns)} features")
    return result


def train_test_split_temporal(
    df: pd.DataFrame,
    test_weeks: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally - last N weeks for testing.
    """
    date_col = 'week_end_date' if 'week_end_date' in df.columns else 'week_sat'

    unique_dates = sorted(df[date_col].unique())
    cutoff_date = unique_dates[-test_weeks]

    train = df[df[date_col] < cutoff_date].copy()
    test = df[df[date_col] >= cutoff_date].copy()

    logger.info(f"Train: {len(train):,} records, Test: {len(test):,} records")
    logger.info(f"Test period: {test[date_col].min()} to {test[date_col].max()}")

    return train, test


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> ModelResults:
    """Calculate evaluation metrics."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    return ModelResults(
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        mape=mape,
        predictions=y_pred,
        actuals=y_true
    )


# ============================================================================
# BASELINE MODELS
# ============================================================================

def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> ModelResults:
    """
    Naive baseline: predict last known value.
    """
    predictions = []
    actuals = []

    for state in test['state'].unique():
        state_train = train[train['state'] == state]
        state_test = test[test['state'] == state]

        if len(state_train) == 0:
            continue

        last_value = state_train[target_col].iloc[-1]

        for _, row in state_test.iterrows():
            predictions.append(last_value)
            actuals.append(row[target_col])

    return evaluate_predictions(
        np.array(actuals),
        np.array(predictions),
        "Naive (Last Value)"
    )


def seasonal_naive_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    season_length: int = 52
) -> ModelResults:
    """
    Seasonal naive: predict value from same week last year.
    Falls back to naive if not enough history.
    """
    date_col = 'week_end_date' if 'week_end_date' in train.columns else 'week_sat'
    predictions = []
    actuals = []

    for state in test['state'].unique():
        state_train = train[train['state'] == state].copy()
        state_test = test[test['state'] == state].copy()

        if len(state_train) == 0:
            continue

        state_train['week_of_year'] = pd.to_datetime(state_train[date_col]).dt.isocalendar().week.astype(int)
        state_test['week_of_year'] = pd.to_datetime(state_test[date_col]).dt.isocalendar().week.astype(int)

        # Create lookup for seasonal values
        seasonal_values = state_train.groupby('week_of_year')[target_col].mean().to_dict()
        last_value = state_train[target_col].iloc[-1]

        for _, row in state_test.iterrows():
            week = row['week_of_year']
            pred = seasonal_values.get(week, last_value)
            predictions.append(pred)
            actuals.append(row[target_col])

    return evaluate_predictions(
        np.array(actuals),
        np.array(predictions),
        "Seasonal Naive"
    )


def arima_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    use_wastewater: bool = True
) -> ModelResults:
    """
    ARIMA model with optional wastewater covariates.
    Uses auto-selection of (p,d,q) parameters.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    date_col = 'week_end_date' if 'week_end_date' in train.columns else 'week_sat'
    predictions = []
    actuals = []

    model_name = "ARIMA + WW" if use_wastewater else "ARIMA"

    for state in test['state'].unique():
        state_train = train[train['state'] == state].sort_values(date_col)
        state_test = test[test['state'] == state].sort_values(date_col)

        if len(state_train) < 10:
            continue

        y_train = state_train[target_col].ffill().fillna(0).values

        # Prepare exogenous variables if using wastewater
        exog_train = None
        exog_test = None
        if use_wastewater and 'percentile' in state_train.columns:
            exog_train = state_train[['percentile']].ffill().fillna(50).values
            exog_test = state_test[['percentile']].ffill().fillna(50).values

        try:
            # Fit ARIMA(1,1,1) - simple but effective
            if exog_train is not None:
                model = ARIMA(y_train, order=(1, 1, 1), exog=exog_train)
            else:
                model = ARIMA(y_train, order=(1, 1, 1))

            fitted = model.fit()

            # Forecast
            n_forecast = len(state_test)
            if exog_test is not None:
                forecast = fitted.forecast(steps=n_forecast, exog=exog_test)
            else:
                forecast = fitted.forecast(steps=n_forecast)

            predictions.extend(forecast)
            actuals.extend(state_test[target_col].values)

        except Exception as e:
            logger.warning(f"ARIMA failed for {state}: {e}")
            # Fall back to last value
            last_val = y_train[-1]
            predictions.extend([last_val] * len(state_test))
            actuals.extend(state_test[target_col].values)

    return evaluate_predictions(
        np.array(actuals),
        np.array(predictions),
        model_name
    )


def xgboost_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    use_wastewater: bool = True
) -> tuple[ModelResults, any]:
    """
    XGBoost model with engineered features.
    Returns both results and the trained model.
    """
    import xgboost as xgb

    # Define feature columns
    feature_cols = []

    # Lagged target features
    for lag in [1, 2, 3, 4]:
        col = f'{target_col}_lag{lag}'
        if col in train.columns:
            feature_cols.append(col)

    # Rolling features
    for col in [f'{target_col}_roll2_mean', f'{target_col}_roll4_mean']:
        if col in train.columns:
            feature_cols.append(col)

    # Wastewater features
    if use_wastewater:
        ww_features = ['percentile', 'percentile_weighted', 'ptc_15d']
        for col in ww_features:
            if col in train.columns:
                feature_cols.append(col)
            # Lagged versions
            for lag in [1, 2]:
                lag_col = f'{col}_lag{lag}'
                if lag_col in train.columns:
                    feature_cols.append(lag_col)

        # Rolling wastewater
        for col in ['percentile_roll2_mean', 'percentile_roll4_mean']:
            if col in train.columns:
                feature_cols.append(col)

    # Time features
    for col in ['week_of_year', 'month']:
        if col in train.columns:
            feature_cols.append(col)

    logger.info(f"XGBoost using {len(feature_cols)} features: {feature_cols}")

    # Prepare data
    train_clean = train.dropna(subset=feature_cols + [target_col])
    test_clean = test.dropna(subset=feature_cols)

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predict
    predictions = model.predict(X_test)

    model_name = "XGBoost + WW" if use_wastewater else "XGBoost (no WW)"
    results = evaluate_predictions(y_test, predictions, model_name)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 10 Feature Importances:\n{importance.head(10)}")

    return results, model, importance


def run_all_baselines(
    data_path: Path,
    target_col: str = 'covid',
    test_weeks: int = 8
) -> pd.DataFrame:
    """
    Run all baseline models and compare results.
    """
    # Load data
    df = load_data(data_path)

    # IMPORTANT: Split data BEFORE creating features to prevent data leakage.
    # Creating lag/rolling features on the full dataset would leak future
    # information into training data.
    train_raw, test_raw = train_test_split_temporal(df, test_weeks)

    # Create features separately for train and test to avoid leakage
    train = create_features(train_raw, target_col)
    test = create_features(test_raw, target_col)

    # Run baselines
    results = []

    print("\n" + "="*70)
    print("RUNNING BASELINE MODELS")
    print("="*70)

    # 1. Naive baseline
    print("\n[1/5] Naive baseline...")
    naive_result = naive_baseline(train, test, target_col)
    results.append(naive_result)
    print(f"      MAE: {naive_result.mae:.2f}, RMSE: {naive_result.rmse:.2f}, MAPE: {naive_result.mape:.1f}%")

    # 2. Seasonal naive
    print("\n[2/5] Seasonal naive baseline...")
    seasonal_result = seasonal_naive_baseline(train, test, target_col)
    results.append(seasonal_result)
    print(f"      MAE: {seasonal_result.mae:.2f}, RMSE: {seasonal_result.rmse:.2f}, MAPE: {seasonal_result.mape:.1f}%")

    # 3. ARIMA without wastewater
    print("\n[3/5] ARIMA (no wastewater)...")
    arima_result = arima_baseline(train, test, target_col, use_wastewater=False)
    results.append(arima_result)
    print(f"      MAE: {arima_result.mae:.2f}, RMSE: {arima_result.rmse:.2f}, MAPE: {arima_result.mape:.1f}%")

    # 4. ARIMA with wastewater
    print("\n[4/5] ARIMA + wastewater covariates...")
    arima_ww_result = arima_baseline(train, test, target_col, use_wastewater=True)
    results.append(arima_ww_result)
    print(f"      MAE: {arima_ww_result.mae:.2f}, RMSE: {arima_ww_result.rmse:.2f}, MAPE: {arima_ww_result.mape:.1f}%")

    # 5. XGBoost without wastewater
    print("\n[5/6] XGBoost (no wastewater)...")
    xgb_result, _, _ = xgboost_baseline(train, test, target_col, use_wastewater=False)
    results.append(xgb_result)
    print(f"      MAE: {xgb_result.mae:.2f}, RMSE: {xgb_result.rmse:.2f}, MAPE: {xgb_result.mape:.1f}%")

    # 6. XGBoost with wastewater
    print("\n[6/6] XGBoost + wastewater features...")
    xgb_ww_result, xgb_model, feature_importance = xgboost_baseline(train, test, target_col, use_wastewater=True)
    results.append(xgb_ww_result)
    print(f"      MAE: {xgb_ww_result.mae:.2f}, RMSE: {xgb_ww_result.rmse:.2f}, MAPE: {xgb_ww_result.mape:.1f}%")

    # Summary table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    summary = pd.DataFrame([
        {
            'Model': r.model_name,
            'MAE': r.mae,
            'RMSE': r.rmse,
            'MAPE (%)': r.mape
        }
        for r in results
    ]).sort_values('MAE')

    print(summary.to_string(index=False))

    # Calculate improvement from wastewater
    arima_improvement = (arima_result.mae - arima_ww_result.mae) / arima_result.mae * 100
    xgb_improvement = (xgb_result.mae - xgb_ww_result.mae) / xgb_result.mae * 100

    print("\n" + "="*70)
    print("WASTEWATER SIGNAL VALUE")
    print("="*70)
    print(f"ARIMA MAE improvement with wastewater:   {arima_improvement:+.1f}%")
    print(f"XGBoost MAE improvement with wastewater: {xgb_improvement:+.1f}%")

    return summary, results, xgb_model, feature_importance


if __name__ == "__main__":
    # Run from project root
    data_path = Path("data/processed/merged_ww_hospital.parquet")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run the data fetching scripts first.")
        exit(1)

    summary, results, model, importance = run_all_baselines(
        data_path,
        target_col='covid',
        test_weeks=8
    )

    # Save results
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    summary.to_csv(output_dir / "baseline_comparison.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    joblib.dump(model, output_dir / "xgboost_baseline.joblib")

    print(f"\nResults saved to {output_dir}/")
