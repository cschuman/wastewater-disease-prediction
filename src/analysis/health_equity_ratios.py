"""
Health Equity Analysis: Hospitalization-to-Wastewater Ratios

This module analyzes whether wastewater surveillance provides differential
value in areas with varying social vulnerability. Based on methodology from:

Expansion of wastewater-based disease surveillance to improve health equity
in California's Central Valley (Frontiers in Public Health, 2023)

Key Hypothesis:
- States with higher SVI will show LOWER case-to-wastewater ratios
  (clinical underreporting due to testing access issues)
- States with higher SVI will show MORE STABLE hospitalization-to-wastewater
  ratios (hospitalizations harder to miss regardless of SVI)

This would demonstrate that wastewater surveillance adds most value in
high-vulnerability areas where clinical surveillance has gaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def get_state_name_to_abbrev() -> dict:
    """Return mapping from full state names to abbreviations."""
    return {
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


def get_latest_file(directory: Path, pattern: str) -> Path:
    """
    Get the most recent file matching a pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files

    Returns:
        Path to the most recent matching file

    Raises:
        FileNotFoundError: If no matching files found
    """
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    return files[-1]  # Return most recent (sorted alphabetically, dates sort correctly)


def load_and_merge_state_data(data_dir: Path) -> pd.DataFrame:
    """Load wastewater and hospitalization data, merge at state-week level."""

    state_map = get_state_name_to_abbrev()

    # Load NWSS wastewater data (get most recent file)
    nwss_dir = data_dir / 'raw' / 'nwss'
    nwss_file = get_latest_file(nwss_dir, 'nwss_metrics_*.parquet')
    nwss = pd.read_parquet(nwss_file)

    # Aggregate wastewater to state-week level
    nwss['week_end_date'] = pd.to_datetime(nwss['date_end'])
    # Convert full state names to abbreviations
    nwss['state'] = nwss['wwtp_jurisdiction'].map(state_map)

    ww_state = nwss.groupby(['state', 'week_end_date']).agg({
        'percentile': 'mean',  # Average wastewater percentile
        'ptc_15d': 'mean',      # Percent change
        'population_served': 'sum',  # Total population covered
        'wwtp_id': 'nunique'   # Number of sites
    }).reset_index()

    ww_state = ww_state.rename(columns={
        'percentile': 'ww_percentile',
        'ptc_15d': 'ww_pct_change',
        'wwtp_id': 'n_ww_sites'
    })

    # Load NHSN hospitalization data (get most recent file)
    nhsn_dir = data_dir / 'raw' / 'nhsn'
    nhsn_file = get_latest_file(nhsn_dir, 'nhsn_weekly_respiratory_*.parquet')
    nhsn = pd.read_parquet(nhsn_file)

    nhsn['week_end_date'] = pd.to_datetime(nhsn['Week Ending Date'])
    nhsn['state'] = nhsn['Geographic aggregation']

    # Extract hospitalization columns
    hosp_cols = {
        'Total Patients Hospitalized with COVID-19': 'covid_hosp',
        'Total Patients Hospitalized with Influenza': 'flu_hosp',
        'Total Patients Hospitalized with RSV': 'rsv_hosp',
        'Total COVID-19 Admissions': 'covid_admits',
        'Total Influenza Admissions': 'flu_admits',
        'Total RSV Admissions': 'rsv_admits'
    }

    hosp_state = nhsn[['state', 'week_end_date'] + list(hosp_cols.keys())].copy()
    hosp_state = hosp_state.rename(columns=hosp_cols)

    # Merge wastewater and hospitalization data
    merged = pd.merge(
        ww_state,
        hosp_state,
        on=['state', 'week_end_date'],
        how='inner'
    )

    # Calculate total respiratory hospitalizations
    merged['respiratory_total'] = (
        merged['covid_hosp'].fillna(0) +
        merged['flu_hosp'].fillna(0) +
        merged['rsv_hosp'].fillna(0)
    )

    print(f"Merged data: {len(merged)} state-weeks")
    print(f"States: {merged['state'].nunique()}")
    print(f"Date range: {merged['week_end_date'].min()} to {merged['week_end_date'].max()}")

    return merged


def calculate_hosp_ww_ratios(df: pd.DataFrame, lag_weeks: int = 1) -> pd.DataFrame:
    """
    Calculate hospitalization-to-wastewater ratios by state.

    Apply lag to account for delay between wastewater signal and hospitalization.
    Higher ratio = more hospitalizations per unit wastewater signal.
    """
    results = []

    for state in df['state'].unique():
        state_df = df[df['state'] == state].sort_values('week_end_date').copy()

        if len(state_df) < 10:
            continue

        # Lag wastewater signal (WW leads hospitalizations)
        state_df['ww_lagged'] = state_df['ww_percentile'].shift(lag_weeks)

        # Drop NaN
        state_df = state_df.dropna(subset=['ww_lagged', 'covid_hosp'])

        if len(state_df) < 5 or state_df['ww_lagged'].std() < 1:
            continue

        # Calculate ratio via regression slope
        # Hospitalization = slope * wastewater + intercept
        # slope = how many hospitalizations per unit wastewater
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            state_df['ww_lagged'],
            state_df['covid_hosp']
        )

        # Also calculate simple ratio of means
        mean_ratio = state_df['covid_hosp'].mean() / state_df['ww_lagged'].mean()

        # Calculate ratio variability (coefficient of variation of weekly ratios)
        state_df['weekly_ratio'] = state_df['covid_hosp'] / state_df['ww_lagged'].replace(0, np.nan)
        ratio_cv = state_df['weekly_ratio'].std() / state_df['weekly_ratio'].mean()

        results.append({
            'state': state,
            'hosp_ww_slope': slope,
            'hosp_ww_r': r_value,
            'hosp_ww_r2': r_value**2,
            'hosp_ww_pvalue': p_value,
            'hosp_ww_mean_ratio': mean_ratio,
            'hosp_ww_ratio_cv': ratio_cv,  # Lower = more stable
            'n_weeks': len(state_df),
            'mean_hosp': state_df['covid_hosp'].mean(),
            'mean_ww': state_df['ww_lagged'].mean()
        })

    return pd.DataFrame(results)


def get_state_svi_rankings() -> pd.DataFrame:
    """
    Return state-level SVI rankings.

    These are population-weighted averages from CDC SVI 2022 county data.
    Higher SVI = more vulnerable.

    Source: CDC/ATSDR Social Vulnerability Index 2022
    https://www.atsdr.cdc.gov/placeandhealth/svi/
    """
    # State-level SVI scores (population-weighted average of county scores)
    # Scale 0-1, higher = more vulnerable
    # Based on CDC SVI 2022 data
    state_svi = {
        'AL': 0.67, 'AK': 0.39, 'AZ': 0.58, 'AR': 0.65, 'CA': 0.54,
        'CO': 0.38, 'CT': 0.40, 'DE': 0.45, 'DC': 0.51, 'FL': 0.55,
        'GA': 0.59, 'HI': 0.41, 'ID': 0.44, 'IL': 0.50, 'IN': 0.52,
        'IA': 0.35, 'KS': 0.43, 'KY': 0.61, 'LA': 0.71, 'ME': 0.42,
        'MD': 0.43, 'MA': 0.37, 'MI': 0.53, 'MN': 0.34, 'MS': 0.75,
        'MO': 0.53, 'MT': 0.44, 'NE': 0.36, 'NV': 0.55, 'NH': 0.28,
        'NJ': 0.45, 'NM': 0.68, 'NY': 0.51, 'NC': 0.55, 'ND': 0.35,
        'OH': 0.52, 'OK': 0.62, 'OR': 0.46, 'PA': 0.47, 'RI': 0.43,
        'SC': 0.60, 'SD': 0.40, 'TN': 0.58, 'TX': 0.58, 'UT': 0.33,
        'VT': 0.32, 'VA': 0.42, 'WA': 0.43, 'WV': 0.66, 'WI': 0.38,
        'WY': 0.38, 'PR': 0.82, 'VI': 0.70, 'GU': 0.65, 'AS': 0.72
    }

    return pd.DataFrame([
        {'state': k, 'svi_score': v}
        for k, v in state_svi.items()
    ])


def analyze_equity_patterns(
    ratios_df: pd.DataFrame,
    svi_df: pd.DataFrame
) -> dict:
    """
    Analyze whether hospitalization-to-wastewater ratios vary by SVI.

    Key questions:
    1. Do high-SVI states have different hosp-to-WW ratios?
    2. Do high-SVI states have more stable or less stable ratios?
    3. Is the WW-hospitalization correlation different by SVI?
    """
    # Merge ratios with SVI
    merged = pd.merge(ratios_df, svi_df, on='state', how='inner')

    # Create SVI quartiles
    merged['svi_quartile'] = pd.qcut(
        merged['svi_score'],
        q=4,
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    )

    results = {
        'data': merged,
        'correlations': {},
        'quartile_comparison': {},
        'interpretation': []
    }

    # Correlation between SVI and various metrics
    for metric in ['hosp_ww_slope', 'hosp_ww_r2', 'hosp_ww_ratio_cv']:
        corr, pval = stats.pearsonr(merged['svi_score'], merged[metric])
        results['correlations'][metric] = {'correlation': corr, 'p_value': pval}

    # Compare metrics by SVI quartile
    for metric in ['hosp_ww_slope', 'hosp_ww_r2', 'hosp_ww_ratio_cv']:
        quartile_means = merged.groupby('svi_quartile')[metric].agg(['mean', 'std', 'count'])
        results['quartile_comparison'][metric] = quartile_means.to_dict()

        # ANOVA test
        groups = [group[metric].values for name, group in merged.groupby('svi_quartile')]
        f_stat, anova_p = stats.f_oneway(*groups)
        results['quartile_comparison'][f'{metric}_anova_p'] = anova_p

    # Interpretation
    svi_r2_corr = results['correlations']['hosp_ww_r2']['correlation']
    svi_cv_corr = results['correlations']['hosp_ww_ratio_cv']['correlation']

    if svi_r2_corr > 0.2:
        results['interpretation'].append(
            f"FINDING: Higher SVI states show STRONGER WW-hospitalization correlation (r={svi_r2_corr:.2f}). "
            "This suggests wastewater is a better predictor in high-vulnerability areas."
        )
    elif svi_r2_corr < -0.2:
        results['interpretation'].append(
            f"FINDING: Higher SVI states show WEAKER WW-hospitalization correlation (r={svi_r2_corr:.2f}). "
            "Wastewater may be less predictive in high-vulnerability areas."
        )
    else:
        results['interpretation'].append(
            f"No strong relationship between SVI and WW-hospitalization correlation (r={svi_r2_corr:.2f})."
        )

    if svi_cv_corr > 0.2:
        results['interpretation'].append(
            f"Higher SVI states show MORE VARIABLE hosp-to-WW ratios (r={svi_cv_corr:.2f}). "
            "Less reliable signal in vulnerable areas."
        )
    elif svi_cv_corr < -0.2:
        results['interpretation'].append(
            f"Higher SVI states show MORE STABLE hosp-to-WW ratios (r={svi_cv_corr:.2f}). "
            "More reliable signal in vulnerable areas."
        )

    return results


def run_equity_analysis(data_dir: Path = Path("data")) -> dict:
    """Run the complete health equity ratio analysis."""

    print("=" * 70)
    print("HEALTH EQUITY ANALYSIS: Hospitalization-to-Wastewater Ratios")
    print("=" * 70)

    # Load and merge data
    print("\n1. Loading data...")
    merged = load_and_merge_state_data(data_dir)

    # Calculate ratios by state
    print("\n2. Calculating hospitalization-to-wastewater ratios...")
    ratios = calculate_hosp_ww_ratios(merged, lag_weeks=1)
    print(f"   Calculated ratios for {len(ratios)} states")

    # Get SVI data
    print("\n3. Loading state SVI data...")
    svi = get_state_svi_rankings()

    # Analyze equity patterns
    print("\n4. Analyzing equity patterns...")
    results = analyze_equity_patterns(ratios, svi)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Correlation: SVI vs WW-Hospitalization Metrics ---")
    for metric, vals in results['correlations'].items():
        print(f"  {metric}: r={vals['correlation']:.3f}, p={vals['p_value']:.4f}")

    print("\n--- Metrics by SVI Quartile ---")
    data = results['data']
    quartile_summary = data.groupby('svi_quartile').agg({
        'hosp_ww_r2': 'mean',
        'hosp_ww_ratio_cv': 'mean',
        'hosp_ww_slope': 'mean',
        'state': 'count'
    }).round(3)
    quartile_summary.columns = ['WW-Hosp R²', 'Ratio CV', 'Slope', 'N States']
    print(quartile_summary.to_string())

    print("\n--- Key Findings ---")
    for finding in results['interpretation']:
        print(f"  • {finding}")

    return results


if __name__ == "__main__":
    results = run_equity_analysis()
