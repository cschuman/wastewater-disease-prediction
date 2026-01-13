"""
Urban/Rural Confounding Analysis

Tests whether the SVI-monitoring gap is confounded by urbanization.

CRITICAL QUESTION:
    Does the infrastructure equity gap persist after accounting for urban/rural differences?

APPROACH:
    1. Download USDA Rural-Urban Continuum Codes (RUCC) 2023
    2. Merge with county-level SVI and wastewater monitoring data
    3. Run regression WITH and WITHOUT urban/rural controls
    4. Compare coefficients to assess confounding

INTERPRETATION:
    - If SVI effect disappears → confounding explains it (reject equity gap hypothesis)
    - If SVI effect persists → strong evidence of true equity gap (support hypothesis)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def load_rucc_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    """
    Load and process USDA Rural-Urban Continuum Codes.

    RUCC Codes:
        Metro counties:
            1 - Metro areas of 1 million+ population
            2 - Metro areas of 250,000 to 1 million
            3 - Metro areas of <250,000 population

        Nonmetro counties:
            4 - Urban population of 20,000+, adjacent to metro
            5 - Urban population of 20,000+, not adjacent to metro
            6 - Urban population of 5,000-20,000, adjacent to metro
            7 - Urban population of 5,000-20,000, not adjacent to metro
            8 - Urban population <5,000, adjacent to metro
            9 - Urban population <5,000, not adjacent to metro
    """
    rucc = pd.read_csv(data_dir / 'external' / 'rucc_2023.csv', encoding='latin-1')

    # Pivot to wide format
    rucc_wide = rucc.pivot(index='FIPS', columns='Attribute', values='Value').reset_index()

    # Clean column names
    rucc_wide.columns = ['fips', 'description', 'population_2020', 'rucc_2023']

    # Zero-pad FIPS to 5 digits
    rucc_wide['fips'] = rucc_wide['fips'].astype(str).str.zfill(5)
    rucc_wide['rucc_2023'] = pd.to_numeric(rucc_wide['rucc_2023'], errors='coerce')
    rucc_wide['population_2020'] = pd.to_numeric(rucc_wide['population_2020'], errors='coerce')

    # Create urban/rural classifications
    rucc_wide['is_metro'] = rucc_wide['rucc_2023'] <= 3  # Metro = 1-3
    rucc_wide['is_rural'] = rucc_wide['rucc_2023'] >= 7  # Most rural = 7-9

    # Metro size categories
    rucc_wide['metro_size'] = rucc_wide['rucc_2023'].map({
        1: 'Large Metro (1M+)',
        2: 'Medium Metro (250k-1M)',
        3: 'Small Metro (<250k)',
        4: 'Nonmetro Urban 20k+',
        5: 'Nonmetro Urban 20k+',
        6: 'Nonmetro Urban 5-20k',
        7: 'Nonmetro Urban 5-20k',
        8: 'Nonmetro Rural',
        9: 'Nonmetro Rural'
    })

    # Simplified 3-category classification
    rucc_wide['urbanization'] = rucc_wide['rucc_2023'].map({
        1: 'Urban', 2: 'Urban', 3: 'Urban',
        4: 'Mixed', 5: 'Mixed', 6: 'Mixed',
        7: 'Rural', 8: 'Rural', 9: 'Rural'
    })

    print(f"Loaded RUCC data for {len(rucc_wide)} counties")
    print(f"\nUrbanization distribution:")
    print(rucc_wide['urbanization'].value_counts())

    return rucc_wide


def load_county_svi(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load county-level SVI data."""
    svi = pd.read_csv(data_dir / 'external' / 'svi_2022_county.csv', encoding='utf-8-sig')

    svi['fips'] = svi['FIPS'].astype(str).str.zfill(5)

    svi_clean = svi[[
        'fips', 'ST_ABBR', 'COUNTY', 'E_TOTPOP',
        'RPL_THEMES', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4'
    ]].copy()

    svi_clean.columns = [
        'fips', 'state', 'county_name', 'population',
        'svi_overall', 'svi_socioeconomic', 'svi_household', 'svi_minority', 'svi_housing'
    ]

    svi_clean = svi_clean[svi_clean['svi_overall'] >= 0].copy()

    return svi_clean


def load_county_wastewater(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load and aggregate wastewater monitoring data to county level."""
    # Try parquet first, fall back to CSV
    nwss_path = data_dir / 'raw' / 'nwss' / 'nwss_metrics_20260111.parquet'
    if nwss_path.exists():
        nwss = pd.read_parquet(nwss_path)
    else:
        nwss = pd.read_csv(data_dir / 'raw' / 'nwss' / 'nwss_metrics_20260111.csv')

    # Handle multi-county sewersheds
    nwss['fips'] = nwss['county_fips'].str.split(',').str[0].str.strip()
    nwss['fips'] = nwss['fips'].str.zfill(5)

    # Aggregate to county level
    county_ww = nwss.groupby('fips').agg({
        'wwtp_id': 'nunique',
        'population_served': 'sum',
    }).reset_index()

    county_ww.columns = ['fips', 'n_sites', 'pop_served']

    return county_ww


def create_analysis_dataset(data_dir: Path = Path("data")) -> pd.DataFrame:
    """
    Merge SVI, RUCC, and wastewater data.

    This creates the complete dataset for confounding analysis.
    """
    print("="*70)
    print("BUILDING ANALYSIS DATASET")
    print("="*70)

    # Load all datasets
    svi = load_county_svi(data_dir)
    rucc = load_rucc_data(data_dir)
    ww = load_county_wastewater(data_dir)

    # Merge SVI + RUCC (should be near-complete match)
    df = pd.merge(svi, rucc[['fips', 'rucc_2023', 'is_metro', 'is_rural',
                               'metro_size', 'urbanization', 'population_2020']],
                  on='fips', how='left')

    print(f"\nAfter SVI + RUCC merge: {len(df)} counties")
    print(f"  Counties with RUCC codes: {df['rucc_2023'].notna().sum()} ({df['rucc_2023'].notna().mean()*100:.1f}%)")

    # Merge with wastewater data
    df = pd.merge(df, ww, on='fips', how='left')

    # Create monitoring indicators
    df['has_monitoring'] = df['n_sites'].notna() & (df['n_sites'] > 0)
    df['n_sites'] = df['n_sites'].fillna(0)

    # Sites per capita (per 100k population)
    df['sites_per_100k'] = df['n_sites'] / (df['population'] / 100000)
    df['sites_per_100k'] = df['sites_per_100k'].replace([np.inf, -np.inf], np.nan)

    # Coverage rate
    df['coverage_pct'] = (df['pop_served'] / df['population'] * 100).clip(upper=100)

    # Create SVI quartiles
    df['svi_quartile'] = pd.qcut(df['svi_overall'], q=4,
                                   labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                                   duplicates='drop')

    print(f"\nFinal analysis dataset: {len(df)} counties")
    print(f"  With monitoring: {df['has_monitoring'].sum()} ({df['has_monitoring'].mean()*100:.1f}%)")
    print(f"  With RUCC codes: {df['rucc_2023'].notna().sum()}")

    return df


def analyze_svi_urbanization_correlation(df: pd.DataFrame) -> dict:
    """
    Test whether SVI is correlated with urbanization.

    If they're highly correlated, we NEED to control for urbanization to isolate SVI effect.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Are SVI and Urbanization Correlated?")
    print("="*70)
    print("(If yes, urbanization could be a confounder)")

    results = {}

    # Filter to counties with both measures
    analysis_df = df.dropna(subset=['svi_overall', 'rucc_2023'])

    # Correlation between SVI and RUCC
    corr, corr_p = stats.pearsonr(analysis_df['svi_overall'], analysis_df['rucc_2023'])

    print(f"\nPearson correlation: SVI vs RUCC code")
    print(f"  r = {corr:.3f}, p = {corr_p:.4f}")

    if corr > 0:
        print(f"  → Higher SVI correlates with higher RUCC (more rural)")
    else:
        print(f"  → Higher SVI correlates with lower RUCC (more urban)")

    results['svi_rucc_corr'] = corr
    results['svi_rucc_p'] = corr_p

    # Mean SVI by urbanization category
    print(f"\n{'-'*50}")
    print("Mean SVI by urbanization:")
    svi_by_urban = analysis_df.groupby('urbanization')['svi_overall'].agg(['mean', 'count'])
    print(svi_by_urban.to_string())

    # ANOVA: Does SVI differ across urban/rural?
    groups = [analysis_df[analysis_df['urbanization'] == cat]['svi_overall'].dropna()
              for cat in ['Urban', 'Mixed', 'Rural']
              if cat in analysis_df['urbanization'].values]

    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
        print(f"\nANOVA: SVI across urbanization categories")
        print(f"  F = {f_stat:.2f}, p = {anova_p:.4f}")
        results['anova_f'] = f_stat
        results['anova_p'] = anova_p

        if anova_p < 0.05:
            print(f"  ✓ SVI DIFFERS significantly by urbanization")
            print(f"    → Urbanization IS a potential confounder")
        else:
            print(f"  No significant difference")

    return results


def analyze_baseline_svi_effect(df: pd.DataFrame) -> dict:
    """
    BASELINE MODEL: SVI effect WITHOUT controlling for urbanization.

    This replicates the original finding.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Baseline SVI Effect (NO urban/rural control)")
    print("="*70)

    results = {}

    # Use only monitored counties with valid data
    analysis_df = df[df['has_monitoring']].copy()
    analysis_df = analysis_df.dropna(subset=['sites_per_100k', 'svi_overall'])

    print(f"\nSample: {len(analysis_df)} counties with monitoring")

    # Simple regression: sites_per_100k ~ SVI
    X = sm.add_constant(analysis_df['svi_overall'])
    y = analysis_df['sites_per_100k']

    baseline_model = sm.OLS(y, X).fit()

    print(f"\nModel: Sites per 100k ~ SVI")
    print(f"  SVI coefficient: {baseline_model.params['svi_overall']:.3f}")
    print(f"  Std error: {baseline_model.bse['svi_overall']:.3f}")
    print(f"  p-value: {baseline_model.pvalues['svi_overall']:.4f}")
    print(f"  R²: {baseline_model.rsquared:.3f}")

    # Interpretation
    if baseline_model.pvalues['svi_overall'] < 0.05:
        direction = "FEWER" if baseline_model.params['svi_overall'] < 0 else "MORE"
        print(f"\n  ✓ SIGNIFICANT: High-SVI counties have {direction} monitoring sites")
        print(f"    Effect size: {abs(baseline_model.params['svi_overall']):.2f} sites per 100k per 1-unit SVI increase")
    else:
        print(f"\n  No significant SVI effect")

    results['baseline_coef'] = baseline_model.params['svi_overall']
    results['baseline_se'] = baseline_model.bse['svi_overall']
    results['baseline_p'] = baseline_model.pvalues['svi_overall']
    results['baseline_r2'] = baseline_model.rsquared
    results['baseline_n'] = len(analysis_df)

    return results, baseline_model


def analyze_controlled_svi_effect(df: pd.DataFrame) -> dict:
    """
    CONTROLLED MODEL: SVI effect WITH urban/rural control.

    This is the KEY analysis - does SVI effect persist after controlling for urbanization?
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: SVI Effect WITH Urban/Rural Control")
    print("="*70)

    results = {}

    # Use only monitored counties with valid data
    analysis_df = df[df['has_monitoring']].copy()
    analysis_df = analysis_df.dropna(subset=['sites_per_100k', 'svi_overall', 'rucc_2023'])

    print(f"\nSample: {len(analysis_df)} counties with monitoring and RUCC data")

    # Model 1: Control with RUCC as continuous variable
    print(f"\n{'='*50}")
    print("Model 1: Sites per 100k ~ SVI + RUCC")

    X1 = sm.add_constant(analysis_df[['svi_overall', 'rucc_2023']])
    y = analysis_df['sites_per_100k']

    model1 = sm.OLS(y, X1).fit()

    print(f"\n  SVI coefficient: {model1.params['svi_overall']:.3f}")
    print(f"  SVI std error: {model1.bse['svi_overall']:.3f}")
    print(f"  SVI p-value: {model1.pvalues['svi_overall']:.4f}")
    print(f"\n  RUCC coefficient: {model1.params['rucc_2023']:.3f}")
    print(f"  RUCC p-value: {model1.pvalues['rucc_2023']:.4f}")
    print(f"\n  R²: {model1.rsquared:.3f}")

    results['model1_svi_coef'] = model1.params['svi_overall']
    results['model1_svi_p'] = model1.pvalues['svi_overall']
    results['model1_rucc_coef'] = model1.params['rucc_2023']
    results['model1_rucc_p'] = model1.pvalues['rucc_2023']
    results['model1_r2'] = model1.rsquared

    # Model 2: Control with metro/nonmetro binary
    print(f"\n{'='*50}")
    print("Model 2: Sites per 100k ~ SVI + Metro")

    X2 = analysis_df[['svi_overall']].copy()
    X2['is_metro'] = analysis_df['is_metro'].astype(float)
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()

    print(f"\n  SVI coefficient: {model2.params['svi_overall']:.3f}")
    print(f"  SVI p-value: {model2.pvalues['svi_overall']:.4f}")
    print(f"\n  Metro coefficient: {model2.params['is_metro']:.3f}")
    print(f"  Metro p-value: {model2.pvalues['is_metro']:.4f}")
    print(f"\n  R²: {model2.rsquared:.3f}")

    results['model2_svi_coef'] = model2.params['svi_overall']
    results['model2_svi_p'] = model2.pvalues['svi_overall']
    results['model2_metro_coef'] = model2.params['is_metro']
    results['model2_metro_p'] = model2.pvalues['is_metro']
    results['model2_r2'] = model2.rsquared

    # Model 3: Control with urbanization categories
    print(f"\n{'='*50}")
    print("Model 3: Sites per 100k ~ SVI + Urbanization Categories")

    # Create dummies for urbanization (reference = Urban)
    urban_dummies = pd.get_dummies(analysis_df['urbanization'], prefix='urb', drop_first=True)
    X3 = pd.concat([analysis_df[['svi_overall']], urban_dummies], axis=1)
    X3 = X3.astype(float)  # Ensure all columns are numeric
    X3 = sm.add_constant(X3)

    model3 = sm.OLS(y, X3).fit()

    print(f"\n  SVI coefficient: {model3.params['svi_overall']:.3f}")
    print(f"  SVI p-value: {model3.pvalues['svi_overall']:.4f}")
    print(f"\n  R²: {model3.rsquared:.3f}")

    results['model3_svi_coef'] = model3.params['svi_overall']
    results['model3_svi_p'] = model3.pvalues['svi_overall']
    results['model3_r2'] = model3.rsquared

    return results, model1, model2, model3


def assess_confounding(baseline_results: dict, controlled_results: dict) -> dict:
    """
    Compare baseline vs controlled models to assess confounding.

    KEY METRICS:
        1. Coefficient change: How much did SVI effect change?
        2. Significance change: Did it remain significant?
        3. R² improvement: Does urbanization explain variance?
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Confounding Assessment")
    print("="*70)

    results = {}

    baseline_coef = baseline_results['baseline_coef']
    controlled_coef = controlled_results['model1_svi_coef']  # Using RUCC continuous model

    # Percent change in coefficient
    pct_change = ((controlled_coef - baseline_coef) / abs(baseline_coef)) * 100

    print(f"\nSVI Coefficient Comparison:")
    print(f"  Baseline (no control):     {baseline_coef:.3f}")
    print(f"  Controlled (+ RUCC):       {controlled_coef:.3f}")
    print(f"  Change:                    {controlled_coef - baseline_coef:.3f} ({pct_change:+.1f}%)")

    # R² comparison
    baseline_r2 = baseline_results['baseline_r2']
    controlled_r2 = controlled_results['model1_r2']
    r2_increase = controlled_r2 - baseline_r2

    print(f"\nModel Fit (R²):")
    print(f"  Baseline:                  {baseline_r2:.3f}")
    print(f"  Controlled:                {controlled_r2:.3f}")
    print(f"  Increase:                  {r2_increase:.3f}")

    # Significance check
    baseline_sig = baseline_results['baseline_p'] < 0.05
    controlled_sig = controlled_results['model1_svi_p'] < 0.05

    print(f"\nStatistical Significance:")
    print(f"  Baseline p-value:          {baseline_results['baseline_p']:.4f} {'***' if baseline_sig else ''}")
    print(f"  Controlled p-value:        {controlled_results['model1_svi_p']:.4f} {'***' if controlled_sig else ''}")

    # INTERPRETATION
    print(f"\n{'='*70}")
    print("CONFOUNDING INTERPRETATION")
    print(f"{'='*70}")

    # Rule of thumb: >10% change in coefficient suggests confounding
    substantial_change = abs(pct_change) > 10

    if substantial_change and not controlled_sig:
        conclusion = "FULL CONFOUNDING"
        explanation = """
The SVI effect DISAPPEARS after controlling for urbanization.
→ The apparent equity gap is EXPLAINED by urban/rural differences.
→ High-SVI states tend to be rural, and rural areas have less infrastructure.
→ This is NOT evidence of a true equity gap in wastewater monitoring.
        """
    elif substantial_change and controlled_sig:
        conclusion = "PARTIAL CONFOUNDING"
        explanation = f"""
The SVI effect is REDUCED by {abs(pct_change):.1f}% but REMAINS SIGNIFICANT.
→ Urbanization explains PART of the gap, but NOT all of it.
→ There is still evidence of an equity gap BEYOND urban/rural differences.
→ High-SVI areas have less monitoring even WITHIN the same urbanization level.
        """
    elif not substantial_change and controlled_sig:
        conclusion = "NO CONFOUNDING"
        explanation = """
The SVI effect is UNCHANGED after controlling for urbanization.
→ SVI and urbanization are INDEPENDENT predictors of monitoring.
→ STRONG evidence of a true equity gap in wastewater monitoring.
→ High-SVI areas have less monitoring regardless of urban/rural status.
        """
    else:
        conclusion = "UNCLEAR"
        explanation = "The pattern does not fit clear confounding criteria."

    print(f"\n{conclusion}")
    print(explanation)

    results['conclusion'] = conclusion
    results['pct_change'] = pct_change
    results['r2_increase'] = r2_increase
    results['baseline_sig'] = baseline_sig
    results['controlled_sig'] = controlled_sig

    return results


def create_stratified_analysis(df: pd.DataFrame) -> dict:
    """
    Stratified analysis: Test SVI effect separately for urban vs rural counties.

    If SVI predicts monitoring ONLY in urban or ONLY in rural areas,
    this suggests an interaction rather than confounding.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: Stratified Analysis (Urban vs Rural)")
    print("="*70)

    results = {}

    # Analyze urban counties only
    urban_df = df[(df['has_monitoring']) & (df['is_metro'] == True)].copy()
    urban_df = urban_df.dropna(subset=['sites_per_100k', 'svi_overall'])

    if len(urban_df) >= 30:  # Minimum sample for reliable regression
        print(f"\n{'='*50}")
        print(f"URBAN COUNTIES ONLY (n={len(urban_df)})")

        X_urban = sm.add_constant(urban_df['svi_overall'])
        y_urban = urban_df['sites_per_100k']

        urban_model = sm.OLS(y_urban, X_urban).fit()

        print(f"\n  SVI coefficient: {urban_model.params['svi_overall']:.3f}")
        print(f"  p-value: {urban_model.pvalues['svi_overall']:.4f}")
        print(f"  R²: {urban_model.rsquared:.3f}")

        results['urban_coef'] = urban_model.params['svi_overall']
        results['urban_p'] = urban_model.pvalues['svi_overall']
        results['urban_n'] = len(urban_df)

    # Analyze rural counties only
    rural_df = df[(df['has_monitoring']) & (df['is_metro'] == False)].copy()
    rural_df = rural_df.dropna(subset=['sites_per_100k', 'svi_overall'])

    if len(rural_df) >= 30:
        print(f"\n{'='*50}")
        print(f"RURAL COUNTIES ONLY (n={len(rural_df)})")

        X_rural = sm.add_constant(rural_df['svi_overall'])
        y_rural = rural_df['sites_per_100k']

        rural_model = sm.OLS(y_rural, X_rural).fit()

        print(f"\n  SVI coefficient: {rural_model.params['svi_overall']:.3f}")
        print(f"  p-value: {rural_model.pvalues['svi_overall']:.4f}")
        print(f"  R²: {rural_model.rsquared:.3f}")

        results['rural_coef'] = rural_model.params['svi_overall']
        results['rural_p'] = rural_model.pvalues['svi_overall']
        results['rural_n'] = len(rural_df)

    # Test for interaction
    if 'urban_coef' in results and 'rural_coef' in results:
        print(f"\n{'='*50}")
        print("INTERACTION TEST")

        coef_diff = results['urban_coef'] - results['rural_coef']
        print(f"\n  Urban SVI effect:  {results['urban_coef']:.3f}")
        print(f"  Rural SVI effect:  {results['rural_coef']:.3f}")
        print(f"  Difference:        {coef_diff:.3f}")

        if abs(coef_diff) > 0.5:  # Arbitrary threshold
            print(f"\n  → SVI effect DIFFERS between urban and rural")
            print(f"    This suggests an INTERACTION, not simple confounding")

    return results


def run_full_confounding_analysis(data_dir: Path = Path("data")) -> dict:
    """
    Complete urban/rural confounding analysis.

    Returns comprehensive results dictionary.
    """
    print("="*70)
    print("URBAN/RURAL CONFOUNDING ANALYSIS")
    print("="*70)
    print("Testing whether SVI-monitoring gap is confounded by urbanization")

    # Build dataset
    df = create_analysis_dataset(data_dir)

    # Save merged dataset for future use
    output_path = data_dir / 'processed' / 'county_svi_rucc_monitoring.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset to: {output_path}")

    all_results = {
        'n_counties': len(df),
        'n_monitored': df['has_monitoring'].sum()
    }

    # Analysis 1: SVI-urbanization correlation
    corr_results = analyze_svi_urbanization_correlation(df)
    all_results['correlation'] = corr_results

    # Analysis 2: Baseline SVI effect (no control)
    baseline_results, baseline_model = analyze_baseline_svi_effect(df)
    all_results['baseline'] = baseline_results

    # Analysis 3: Controlled SVI effect (with urban/rural)
    controlled_results, model1, model2, model3 = analyze_controlled_svi_effect(df)
    all_results['controlled'] = controlled_results

    # Analysis 4: Confounding assessment
    confounding_results = assess_confounding(baseline_results, controlled_results)
    all_results['confounding'] = confounding_results

    # Analysis 5: Stratified analysis
    stratified_results = create_stratified_analysis(df)
    all_results['stratified'] = stratified_results

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    conclusion = confounding_results['conclusion']

    print(f"""
RESEARCH QUESTION:
    Does the SVI-monitoring gap persist after controlling for urbanization?

ANSWER: {conclusion}

KEY FINDINGS:
    1. SVI-RUCC correlation: r = {corr_results.get('svi_rucc_corr', 'N/A'):.3f}
       {"→ High-SVI counties tend to be more RURAL" if corr_results.get('svi_rucc_corr', 0) > 0 else "→ High-SVI counties tend to be more URBAN"}

    2. Baseline SVI effect: {baseline_results['baseline_coef']:.3f} (p={baseline_results['baseline_p']:.4f})
       {"→ SIGNIFICANT negative effect" if baseline_results['baseline_p'] < 0.05 and baseline_results['baseline_coef'] < 0 else ""}

    3. Controlled SVI effect: {controlled_results['model1_svi_coef']:.3f} (p={controlled_results['model1_svi_p']:.4f})
       {"→ REMAINS significant after controlling for RUCC" if controlled_results['model1_svi_p'] < 0.05 else "→ No longer significant"}

    4. Coefficient change: {confounding_results['pct_change']:+.1f}%
       {"→ Substantial change, suggesting confounding" if abs(confounding_results['pct_change']) > 10 else "→ Minimal change"}

    5. R² improvement: {confounding_results['r2_increase']:.3f}
       {"→ Urbanization explains additional variance" if confounding_results['r2_increase'] > 0.05 else "→ Urbanization adds little explanatory power"}

IMPLICATION FOR EQUITY ANALYSIS:
    {confounding_results['conclusion']}

    {
        "The infrastructure equity gap is REAL and persists beyond urban/rural differences."
        if confounding_results['conclusion'] in ['NO CONFOUNDING', 'PARTIAL CONFOUNDING']
        else "The apparent equity gap may be explained by urban/rural infrastructure patterns."
    }
""")

    return all_results, df


if __name__ == "__main__":
    # Run analysis from project root
    data_dir = Path(__file__).parent.parent.parent / "data"
    results, df = run_full_confounding_analysis(data_dir)
