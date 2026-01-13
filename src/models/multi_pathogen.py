"""
Multi-Pathogen Respiratory Burden Prediction Model

Uses COVID-19 and Influenza A wastewater signals to predict:
- COVID-19 hospitalizations
- Influenza hospitalizations
- RSV hospitalizations (no wastewater signal available)
- Combined respiratory burden

Usage:
    python -m src.models.multi_pathogen
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# State abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'Puerto Rico': 'PR', 'Virgin Islands': 'VI', 'Guam': 'GU', 'American Samoa': 'AS'
}


def load_and_merge_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge COVID wastewater, Flu wastewater, and hospital data.
    """
    data_dir = Path(data_dir)

    # Load COVID wastewater
    covid_ww_file = list((data_dir / 'raw' / 'nwss').glob('*.parquet'))[0]
    covid_ww = pd.read_parquet(covid_ww_file)
    logger.info(f"Loaded COVID wastewater: {len(covid_ww):,} records")

    # Map state names to abbreviations
    covid_ww['state'] = covid_ww['wwtp_jurisdiction'].map(STATE_ABBREV)
    covid_ww = covid_ww[covid_ww['state'].notna()]

    # Aggregate to state-week
    covid_ww['week_end'] = pd.to_datetime(covid_ww['date_end']).dt.to_period('W-SAT').dt.end_time.dt.normalize()
    covid_ww['pop_weighted_percentile'] = covid_ww['percentile'] * covid_ww['population_served']

    covid_agg = covid_ww.groupby(['state', 'week_end']).agg({
        'percentile': 'mean',
        'pop_weighted_percentile': 'sum',
        'population_served': 'sum',
        'ptc_15d': 'mean',
        'wwtp_id': 'nunique'
    }).reset_index()
    covid_agg['covid_ww_percentile'] = covid_agg['pop_weighted_percentile'] / covid_agg['population_served']
    covid_agg = covid_agg.rename(columns={
        'percentile': 'covid_ww_percentile_raw',
        'ptc_15d': 'covid_ww_ptc',
        'wwtp_id': 'covid_n_sites'
    })
    covid_agg = covid_agg.drop(columns=['pop_weighted_percentile', 'population_served'])

    # Load Flu wastewater
    flu_ww_file = list((data_dir / 'raw' / 'flu_wastewater').glob('*.parquet'))[0]
    flu_ww = pd.read_parquet(flu_ww_file)
    logger.info(f"Loaded Flu wastewater: {len(flu_ww):,} records")

    # Map state names to abbreviations
    flu_ww['state'] = flu_ww['wwtp_jurisdiction'].map(STATE_ABBREV)
    flu_ww = flu_ww[flu_ww['state'].notna()]

    # Aggregate to state-week
    flu_ww['week_end'] = pd.to_datetime(flu_ww['sample_collect_date']).dt.to_period('W-SAT').dt.end_time.dt.normalize()

    flu_agg = flu_ww.groupby(['state', 'week_end']).agg({
        'pcr_target_avg_conc_lin': 'mean',
        'pcr_target_flowpop_lin': 'mean',
        'sewershed_id': 'nunique'
    }).reset_index()
    flu_agg = flu_agg.rename(columns={
        'pcr_target_avg_conc_lin': 'flu_ww_conc',
        'pcr_target_flowpop_lin': 'flu_ww_flowpop',
        'sewershed_id': 'flu_n_sites'
    })

    # Load Hospital data
    hosp_file = list((data_dir / 'raw' / 'nhsn').glob('*.parquet'))[0]
    hosp = pd.read_parquet(hosp_file)
    logger.info(f"Loaded Hospital data: {len(hosp):,} records")

    # Filter to states
    state_codes = list(STATE_ABBREV.values())
    hosp = hosp[hosp['Geographic aggregation'].isin(state_codes)].copy()
    hosp['state'] = hosp['Geographic aggregation']
    hosp['week_end'] = pd.to_datetime(hosp['Week Ending Date']).dt.normalize()

    # Extract admission columns
    hosp_clean = hosp[['state', 'week_end',
                       'Total COVID-19 Admissions',
                       'Total Influenza Admissions',
                       'Total RSV Admissions']].copy()
    hosp_clean.columns = ['state', 'week_end', 'covid_hosp', 'flu_hosp', 'rsv_hosp']

    for col in ['covid_hosp', 'flu_hosp', 'rsv_hosp']:
        hosp_clean[col] = pd.to_numeric(hosp_clean[col], errors='coerce')

    hosp_clean['respiratory_total'] = (
        hosp_clean['covid_hosp'].fillna(0) +
        hosp_clean['flu_hosp'].fillna(0) +
        hosp_clean['rsv_hosp'].fillna(0)
    )

    # Merge all datasets
    merged = pd.merge(covid_agg, flu_agg, on=['state', 'week_end'], how='outer')
    merged['week_end_date'] = merged['week_end'].dt.date
    hosp_clean['week_end_date'] = hosp_clean['week_end'].dt.date

    merged = pd.merge(merged, hosp_clean, on=['state', 'week_end_date'], how='inner', suffixes=('', '_hosp'))

    logger.info(f"Merged dataset: {len(merged):,} records, {merged['state'].nunique()} states")

    return merged


def create_multi_pathogen_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features using both COVID and Flu wastewater signals.
    """
    df = df.copy()

    # Sort by state and date
    df = df.sort_values(['state', 'week_end_date'])

    feature_dfs = []

    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()

        # COVID wastewater features
        for col in ['covid_ww_percentile', 'covid_ww_ptc']:
            if col in state_df.columns:
                for lag in [1, 2, 3]:
                    state_df[f'{col}_lag{lag}'] = state_df[col].shift(lag)
                state_df[f'{col}_roll2'] = state_df[col].rolling(2).mean()

        # Flu wastewater features
        for col in ['flu_ww_conc', 'flu_ww_flowpop']:
            if col in state_df.columns:
                for lag in [1, 2, 3]:
                    state_df[f'{col}_lag{lag}'] = state_df[col].shift(lag)
                state_df[f'{col}_roll2'] = state_df[col].rolling(2).mean()

        # Target lags (for each disease)
        for target in ['covid_hosp', 'flu_hosp', 'rsv_hosp', 'respiratory_total']:
            if target in state_df.columns:
                for lag in [1, 2, 3]:
                    state_df[f'{target}_lag{lag}'] = state_df[target].shift(lag)
                state_df[f'{target}_roll2'] = state_df[target].rolling(2).mean()

        # Time features
        state_df['week_of_year'] = pd.to_datetime(state_df['week_end_date']).dt.isocalendar().week.astype(int)
        state_df['month'] = pd.to_datetime(state_df['week_end_date']).dt.month

        feature_dfs.append(state_df)

    result = pd.concat(feature_dfs, ignore_index=True)
    logger.info(f"Created features: {len(result.columns)} columns")

    return result


def train_test_split_temporal(df: pd.DataFrame, test_weeks: int = 8):
    """Split data temporally."""
    unique_dates = sorted(df['week_end_date'].unique())
    cutoff = unique_dates[-test_weeks]

    train = df[df['week_end_date'] < cutoff].copy()
    test = df[df['week_end_date'] >= cutoff].copy()

    logger.info(f"Train: {len(train):,}, Test: {len(test):,}")
    return train, test


def build_xgboost_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    use_covid_ww: bool = True,
    use_flu_ww: bool = True
) -> dict:
    """
    Build XGBoost model for a specific target.

    Args:
        train: Training data
        test: Test data
        target_col: Target column name
        use_covid_ww: Include COVID wastewater features
        use_flu_ww: Include Flu wastewater features

    Returns:
        Dictionary with results, model, and feature importance
    """
    # Define feature columns
    feature_cols = []

    # Target lags
    for lag in [1, 2, 3]:
        col = f'{target_col}_lag{lag}'
        if col in train.columns:
            feature_cols.append(col)

    # Target rolling
    roll_col = f'{target_col}_roll2'
    if roll_col in train.columns:
        feature_cols.append(roll_col)

    # COVID wastewater features
    if use_covid_ww:
        covid_features = ['covid_ww_percentile', 'covid_ww_percentile_lag1', 'covid_ww_percentile_lag2',
                          'covid_ww_ptc', 'covid_ww_percentile_roll2']
        for f in covid_features:
            if f in train.columns and train[f].notna().sum() > 100:  # Only use if enough data
                feature_cols.append(f)

    # Flu wastewater features
    if use_flu_ww:
        flu_features = ['flu_ww_conc', 'flu_ww_conc_lag1', 'flu_ww_conc_lag2',
                        'flu_ww_flowpop', 'flu_ww_conc_roll2']
        for f in flu_features:
            if f in train.columns and train[f].notna().sum() > 100:  # Only use if enough data
                feature_cols.append(f)

    # Time features
    for f in ['week_of_year', 'month']:
        if f in train.columns:
            feature_cols.append(f)

    logger.info(f"Training {target_col} with {len(feature_cols)} features")

    # Prepare data - fill NaN with median for wastewater columns
    train_clean = train.copy()
    test_clean = test.copy()

    for col in feature_cols:
        if col in train_clean.columns:
            median_val = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(median_val)
            test_clean[col] = test_clean[col].fillna(median_val)

    # Drop rows where target is NaN
    train_clean = train_clean.dropna(subset=[target_col])
    test_clean = test_clean.dropna(subset=[target_col])

    if len(train_clean) < 50 or len(test_clean) < 10:
        logger.warning(f"Insufficient data for {target_col}: train={len(train_clean)}, test={len(test_clean)}")
        return None

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values

    # Train model
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

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    nonzero = y_test != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_test[nonzero] - predictions[nonzero]) / y_test[nonzero])) * 100
    else:
        mape = np.nan

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'target': target_col,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'model': model,
        'importance': importance,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def run_multi_pathogen_experiment(data_dir: Path) -> pd.DataFrame:
    """
    Run the complete multi-pathogen modeling experiment.
    """
    # Load and prepare data
    print("\n" + "="*70)
    print("MULTI-PATHOGEN MODEL EXPERIMENT")
    print("="*70)

    merged = load_and_merge_data(data_dir)
    featured = create_multi_pathogen_features(merged)
    train, test = train_test_split_temporal(featured, test_weeks=8)

    # Define experiments
    experiments = [
        # Target, Use COVID WW, Use Flu WW, Description
        ('covid_hosp', False, False, 'COVID (no WW)'),
        ('covid_hosp', True, False, 'COVID (+ COVID WW)'),
        ('covid_hosp', True, True, 'COVID (+ Both WW)'),
        ('flu_hosp', False, False, 'Flu (no WW)'),
        ('flu_hosp', False, True, 'Flu (+ Flu WW)'),
        ('flu_hosp', True, True, 'Flu (+ Both WW)'),
        ('rsv_hosp', False, False, 'RSV (no WW)'),
        ('rsv_hosp', True, True, 'RSV (+ Both WW)'),
        ('respiratory_total', False, False, 'Total (no WW)'),
        ('respiratory_total', True, True, 'Total (+ Both WW)'),
    ]

    results = []

    print("\n" + "-"*70)
    print("TRAINING MODELS")
    print("-"*70)

    for target, use_covid, use_flu, desc in experiments:
        result = build_xgboost_model(train, test, target, use_covid, use_flu)

        if result:
            results.append({
                'Description': desc,
                'Target': target,
                'COVID_WW': use_covid,
                'Flu_WW': use_flu,
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'MAPE': result['mape'],
                'N_Train': result['n_train'],
                'N_Test': result['n_test']
            })
            print(f"{desc:25s} MAE: {result['mae']:8.2f}  RMSE: {result['rmse']:8.2f}  MAPE: {result['mape']:5.1f}%")

    results_df = pd.DataFrame(results)

    # Print summary by target
    print("\n" + "="*70)
    print("RESULTS BY TARGET")
    print("="*70)

    for target in ['covid_hosp', 'flu_hosp', 'rsv_hosp', 'respiratory_total']:
        target_results = results_df[results_df['Target'] == target]
        if len(target_results) > 0:
            print(f"\n{target.upper()}:")
            print(target_results[['Description', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))

            # Calculate improvement
            no_ww = target_results[~target_results['COVID_WW'] & ~target_results['Flu_WW']]['MAE'].values
            with_ww = target_results[target_results['COVID_WW'] | target_results['Flu_WW']]['MAE'].min()

            if len(no_ww) > 0 and not np.isnan(with_ww):
                improvement = (no_ww[0] - with_ww) / no_ww[0] * 100
                print(f"  --> Improvement with wastewater: {improvement:+.1f}%")

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL MODEL COMPARISON")
    print("="*70)
    print(results_df.sort_values('MAE')[['Description', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))

    return results_df


if __name__ == "__main__":
    data_dir = Path("data")

    results = run_multi_pathogen_experiment(data_dir)

    # Save results
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "multi_pathogen_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/multi_pathogen_results.csv")
