"""
Improved forecasting model with better handling of surge periods.

Improvements:
1. Rate of change features (momentum)
2. Acceleration features (detecting surge onset)
3. Wastewater rate of change
4. Ensemble of multiple models
5. Adaptive weighting for recent data
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from pathlib import Path


def create_enhanced_features(df: pd.DataFrame, target: str = 'respiratory_total') -> pd.DataFrame:
    """
    Create enhanced features for better surge prediction.
    """
    df = df.copy()

    # Basic lag features
    for lag in [1, 2, 3, 4]:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    # Rolling statistics
    df[f'{target}_roll2'] = df[target].rolling(2).mean()
    df[f'{target}_roll4'] = df[target].rolling(4).mean()

    # MOMENTUM: Week-over-week change (rate of change)
    df[f'{target}_change1'] = df[target].diff(1)
    # Replace inf values that occur when dividing by zero
    df[f'{target}_pct_change1'] = df[target].pct_change(1).replace([np.inf, -np.inf], np.nan) * 100

    # ACCELERATION: Change in the rate of change
    df[f'{target}_accel'] = df[f'{target}_change1'].diff(1)

    # Lagged momentum (what was the trend last week?)
    df[f'{target}_change1_lag1'] = df[f'{target}_change1'].shift(1)
    df[f'{target}_pct_change1_lag1'] = df[f'{target}_pct_change1'].shift(1)

    # Wastewater momentum
    if 'covid_ww_percentile' in df.columns:
        df['ww_change1'] = df['covid_ww_percentile'].diff(1)
        # Replace inf values that occur when dividing by zero
        df['ww_pct_change1'] = df['covid_ww_percentile'].pct_change(1).replace([np.inf, -np.inf], np.nan) * 100
        df['ww_change1_lag1'] = df['ww_change1'].shift(1)

    # Ratio to recent baseline (detecting deviation from normal)
    # Replace zeros with NaN before division to avoid inf
    roll4_shifted = df[f'{target}_roll4'].shift(1).replace(0, np.nan)
    df[f'{target}_ratio_to_roll4'] = df[target] / roll4_shifted

    # Seasonal features
    df['week_of_year'] = df['week_end_date'].dt.isocalendar().week.astype(int)
    df['month'] = df['week_end_date'].dt.month

    # Winter flag (higher weight on recent data during winter)
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)

    return df


def get_feature_cols(df: pd.DataFrame, target: str = 'respiratory_total') -> list:
    """Get the enhanced feature columns."""
    base_features = [
        f'{target}_lag1', f'{target}_lag2', f'{target}_lag3', f'{target}_lag4',
        f'{target}_roll2', f'{target}_roll4',
        f'{target}_change1', f'{target}_pct_change1',
        f'{target}_accel',
        f'{target}_change1_lag1', f'{target}_pct_change1_lag1',
        f'{target}_ratio_to_roll4',
        'week_of_year', 'month', 'is_winter'
    ]

    # Add wastewater features if available
    ww_features = ['covid_ww_percentile', 'ww_change1', 'ww_pct_change1', 'ww_change1_lag1']

    feature_cols = []
    for f in base_features + ww_features:
        if f in df.columns:
            feature_cols.append(f)

    return feature_cols


class EnsembleForecaster:
    """
    Ensemble forecaster combining multiple models.
    """

    def __init__(self, use_adaptive_weights: bool = True):
        self.use_adaptive_weights = use_adaptive_weights
        self.models = {}
        self.weights = {}

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fit all models in the ensemble."""

        # XGBoost - good at capturing non-linear patterns
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb'].fit(X, y, sample_weight=sample_weight, verbose=False)

        # Gradient Boosting - different regularization
        self.models['gbm'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gbm'].fit(X, y, sample_weight=sample_weight)

        # Random Forest - captures variance
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X, y, sample_weight=sample_weight)

        # Ridge - linear baseline
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X, y, sample_weight=sample_weight)

        # Default equal weights
        self.weights = {'xgb': 0.4, 'gbm': 0.3, 'rf': 0.2, 'ridge': 0.1}

        return self

    def predict(self, X: np.ndarray, recent_momentum: float = None) -> np.ndarray:
        """
        Generate ensemble prediction.

        If recent_momentum is provided and high, give more weight to
        momentum-sensitive models (XGBoost, GBM).
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Adaptive weighting during surges
        weights = self.weights.copy()
        if self.use_adaptive_weights and recent_momentum is not None:
            if abs(recent_momentum) > 20:  # >20% week-over-week change
                # During surges, trust tree-based models more
                weights = {'xgb': 0.5, 'gbm': 0.35, 'rf': 0.1, 'ridge': 0.05}

        # Weighted average
        ensemble_pred = np.zeros_like(predictions['xgb'])
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred

        return ensemble_pred


def create_sample_weights(n_samples: int, decay_factor: float = 0.98) -> np.ndarray:
    """
    Create exponentially decaying sample weights.
    More recent samples get higher weight.
    """
    weights = np.array([decay_factor ** (n_samples - i - 1) for i in range(n_samples)])
    return weights / weights.sum() * n_samples  # Normalize to mean=1


def run_improved_backtest(
    data_dir: Path = Path("data"),
    target: str = "respiratory_total",
    test_start: str = "2024-11-01",
    test_end: str = "2025-03-01",
    horizon: int = 1
) -> dict:
    """
    Run backtest with improved model.
    """
    from src.models.multi_pathogen import load_and_merge_data, create_multi_pathogen_features

    # Load data
    merged = load_and_merge_data(data_dir)
    df = create_multi_pathogen_features(merged)

    # Aggregate to national level
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

    # Create enhanced features
    national = create_enhanced_features(national, target)
    feature_cols = get_feature_cols(national, target)

    # Handle NaN/inf in features
    for col in feature_cols:
        if col in national.columns:
            national[col] = national[col].replace([np.inf, -np.inf], np.nan)
            national[col] = national[col].fillna(national[col].median())

    # Get test period indices
    test_mask = (national['week_end_date'] >= test_start) & (national['week_end_date'] <= test_end)
    test_indices = national[test_mask].index.tolist()

    if len(test_indices) == 0:
        raise ValueError("No data in test period")

    test_start_idx = test_indices[0]
    test_end_idx = test_indices[-1]

    # Run walk-forward backtest
    results = []

    for i in range(test_start_idx, test_end_idx - horizon + 2):
        train_df = national.iloc[:i].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values

        # Create sample weights (recent data weighted more)
        sample_weights = create_sample_weights(len(X_train), decay_factor=0.97)

        # Train ensemble
        ensemble = EnsembleForecaster(use_adaptive_weights=True)
        ensemble.fit(X_train, y_train, sample_weight=sample_weights)

        # Get recent momentum for adaptive prediction
        recent_momentum = train_df[f'{target}_pct_change1'].iloc[-1] if f'{target}_pct_change1' in train_df.columns else 0

        # Predict
        for h in range(horizon):
            pred_idx = i + h
            if pred_idx > test_end_idx:
                break

            X_pred = national.iloc[pred_idx:pred_idx+1][feature_cols].values
            prediction = ensemble.predict(X_pred, recent_momentum=recent_momentum)[0]
            actual = national.iloc[pred_idx][target]
            pred_date = national.iloc[pred_idx]['week_end_date']

            results.append({
                'date': pred_date,
                'actual': actual,
                'predicted': prediction,
                'horizon': h + 1
            })

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['horizon'] == horizon]

    mae = np.mean(np.abs(results_df['actual'] - results_df['predicted']))

    # MAPE calculation with zero protection (matching baseline.py approach)
    nonzero_mask = results_df['actual'] != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(
            (results_df['actual'][nonzero_mask] - results_df['predicted'][nonzero_mask]) /
            results_df['actual'][nonzero_mask]
        )) * 100
    else:
        mape = np.nan

    return {
        'mae': mae,
        'mape': mape,
        'results': results_df
    }


if __name__ == "__main__":
    print("Testing improved forecaster on winter period...")

    for horizon in [1, 2, 3, 4]:
        result = run_improved_backtest(horizon=horizon)
        print(f"{horizon}-week horizon: MAE={result['mae']:,.0f}, MAPE={result['mape']:.1f}%")
