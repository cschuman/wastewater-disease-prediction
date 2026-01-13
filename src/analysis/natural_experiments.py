"""
Natural Experiments Analysis

Search for quasi-experimental variation that can strengthen causal claims:

1. Site Expansion Events: States that added monitoring sites during the study period
2. COVID Surge Intensity: Compare equity gap during high vs low transmission periods
3. Temporal Variation: Did the equity gap change over time?

For difference-in-differences:
- Treatment: States that significantly expanded monitoring
- Control: States with stable site counts
- Pre/Post: Before/after expansion
- Outcome: Signal quality metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


def analyze_site_expansion(data_dir: Path = Path("data")) -> dict:
    """
    Find states that expanded wastewater monitoring during the study period.

    Look for states where:
    - Number of active sites increased significantly
    - New sites came online during 2024-2025
    """
    print("=" * 70)
    print("NATURAL EXPERIMENT 1: Site Expansion Events")
    print("=" * 70)

    # Load wastewater data
    nwss = pd.read_parquet(data_dir / "raw" / "nwss" / "nwss_metrics_20260111.parquet")

    # Get state abbreviations
    state_map = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "Puerto Rico": "PR",
        "Virgin Islands": "VI",
        "Guam": "GU",
        "American Samoa": "AS",
    }
    nwss["state"] = nwss["wwtp_jurisdiction"].map(state_map)
    nwss["week"] = pd.to_datetime(nwss["date_end"])

    # Count active sites per state per quarter
    nwss["quarter"] = nwss["week"].dt.to_period("Q")

    site_counts = nwss.groupby(["state", "quarter"])["wwtp_id"].nunique().reset_index()
    site_counts.columns = ["state", "quarter", "n_sites"]

    # Pivot to compare first vs last quarter
    quarters = sorted(site_counts["quarter"].unique())
    if len(quarters) < 2:
        print("Not enough quarters for comparison")
        return {}

    first_q = quarters[0]
    last_q = quarters[-1]

    print(f"\nComparing {first_q} to {last_q}")

    first_counts = site_counts[site_counts["quarter"] == first_q][["state", "n_sites"]]
    first_counts.columns = ["state", "sites_first"]

    last_counts = site_counts[site_counts["quarter"] == last_q][["state", "n_sites"]]
    last_counts.columns = ["state", "sites_last"]

    comparison = pd.merge(first_counts, last_counts, on="state", how="outer").fillna(0)
    comparison["sites_first"] = comparison["sites_first"].astype(int)
    comparison["sites_last"] = comparison["sites_last"].astype(int)
    comparison["change"] = comparison["sites_last"] - comparison["sites_first"]
    comparison["pct_change"] = (
        comparison["change"] / comparison["sites_first"].replace(0, 1) * 100
    ).round(1)

    # Identify expansion states (>20% increase or >5 new sites)
    comparison["expanded"] = (comparison["pct_change"] > 20) | (comparison["change"] >= 5)
    comparison["contracted"] = (comparison["pct_change"] < -20) | (comparison["change"] <= -5)

    print(f"\n--- Site Changes by State ---")
    print(comparison.sort_values("change", ascending=False).head(15).to_string(index=False))

    expanders = comparison[comparison["expanded"]]["state"].tolist()
    contractors = comparison[comparison["contracted"]]["state"].tolist()
    stable = comparison[(~comparison["expanded"]) & (~comparison["contracted"])]["state"].tolist()

    print(f"\n--- Summary ---")
    print(f"Expanding states (>20% or +5 sites): {len(expanders)}")
    print(f"  {expanders[:10]}{'...' if len(expanders) > 10 else ''}")
    print(f"Contracting states (<-20% or -5 sites): {len(contractors)}")
    print(f"  {contractors[:10]}{'...' if len(contractors) > 10 else ''}")
    print(f"Stable states: {len(stable)}")

    results = {
        "comparison": comparison,
        "expanders": expanders,
        "contractors": contractors,
        "stable": stable,
        "first_quarter": str(first_q),
        "last_quarter": str(last_q),
    }

    return results


def analyze_first_sample_dates(data_dir: Path = Path("data")) -> dict:
    """
    Analyze when sites first came online using first_sample_date field.

    This gives us cleaner identification of NEW monitoring.
    """
    print("\n" + "=" * 70)
    print("NATURAL EXPERIMENT 2: New Site Onboarding")
    print("=" * 70)

    nwss = pd.read_parquet(data_dir / "raw" / "nwss" / "nwss_metrics_20260111.parquet")

    # Get unique sites with their first sample date
    sites = (
        nwss.groupby("wwtp_id")
        .agg(
            {
                "first_sample_date": "first",
                "wwtp_jurisdiction": "first",
                "population_served": "first",
            }
        )
        .reset_index()
    )

    sites["first_sample_date"] = pd.to_datetime(sites["first_sample_date"])

    # Map to state abbreviations
    state_map = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "Puerto Rico": "PR",
        "Virgin Islands": "VI",
        "Guam": "GU",
        "American Samoa": "AS",
    }
    sites["state"] = sites["wwtp_jurisdiction"].map(state_map)

    print(f"\nTotal unique sites: {len(sites)}")
    print(f"Date range: {sites['first_sample_date'].min()} to {sites['first_sample_date'].max()}")

    # Sites that came online during our study period (2024+)
    new_sites = sites[sites["first_sample_date"] >= "2024-01-01"]
    print(f"\nSites that came online in 2024+: {len(new_sites)}")

    # By state
    new_by_state = new_sites.groupby("state").size().sort_values(ascending=False)
    print("\nNew sites (2024+) by state:")
    print(new_by_state.head(15).to_string())

    # Timeline of site additions
    sites["month"] = sites["first_sample_date"].dt.to_period("M")
    monthly = sites.groupby("month").size()

    print("\n--- Monthly Site Additions ---")
    recent = monthly[monthly.index >= "2024-01"]
    print(recent.to_string())

    # Identify states with significant 2024+ expansion
    total_by_state = sites.groupby("state").size()
    new_pct = (new_by_state / total_by_state * 100).fillna(0).sort_values(ascending=False)

    print("\n--- States with highest % of new sites (2024+) ---")
    print(new_pct.head(10).round(1).to_string())

    return {
        "new_sites": new_sites,
        "new_by_state": new_by_state.to_dict(),
        "monthly_additions": monthly.to_dict(),
        "new_pct_by_state": new_pct.to_dict(),
    }


def difference_in_differences(
    data_dir: Path = Path("data"), expansion_results: dict = None
) -> dict:
    """
    Run difference-in-differences analysis.

    Compare signal quality changes between:
    - Treatment: States that expanded monitoring
    - Control: States with stable monitoring

    Before/After: Split at midpoint of study period
    """
    print("\n" + "=" * 70)
    print("NATURAL EXPERIMENT 3: Difference-in-Differences")
    print("=" * 70)

    if expansion_results is None:
        expansion_results = analyze_site_expansion(data_dir)

    if not expansion_results.get("expanders"):
        print("No expanding states found")
        return {}

    # Load data
    nwss = pd.read_parquet(data_dir / "raw" / "nwss" / "nwss_metrics_20260111.parquet")

    state_map = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "Puerto Rico": "PR",
        "Virgin Islands": "VI",
        "Guam": "GU",
        "American Samoa": "AS",
    }
    nwss["state"] = nwss["wwtp_jurisdiction"].map(state_map)
    nwss["week"] = pd.to_datetime(nwss["date_end"])

    # Define treatment groups
    expanders = expansion_results["expanders"]
    stable = expansion_results["stable"]

    # Only use states in either group
    nwss = nwss[nwss["state"].isin(expanders + stable)].copy()

    # Define pre/post periods (split at midpoint)
    midpoint = nwss["week"].min() + (nwss["week"].max() - nwss["week"].min()) / 2
    nwss["post"] = (nwss["week"] >= midpoint).astype(int)
    nwss["treatment"] = nwss["state"].isin(expanders).astype(int)

    print(f"\nMidpoint: {midpoint.date()}")
    print(f"Treatment states (expanders): {len(expanders)}")
    print(f"Control states (stable): {len(stable)}")

    # Aggregate to state-period level
    did_data = (
        nwss.groupby(["state", "post", "treatment"])
        .agg({"percentile": "mean", "wwtp_id": "nunique", "detect_prop_15d": "mean"})
        .reset_index()
    )

    did_data.columns = ["state", "post", "treatment", "ww_signal", "n_sites", "detect_rate"]

    # Calculate signal variability within each state-period
    signal_var = nwss.groupby(["state", "post"]).agg({"percentile": "std"}).reset_index()
    signal_var.columns = ["state", "post", "signal_std"]

    did_data = pd.merge(did_data, signal_var, on=["state", "post"], how="left")

    # DiD regression: outcome ~ treatment + post + treatment*post
    print("\n--- Difference-in-Differences Regression ---")
    print("Model: Signal_Std ~ Treatment + Post + Treatment*Post")

    did_data["treat_post"] = did_data["treatment"] * did_data["post"]

    y = did_data["signal_std"].dropna()
    X = did_data.loc[y.index, ["treatment", "post", "treat_post"]]
    X = sm.add_constant(X)

    did_model = sm.OLS(y, X).fit()

    print(f"\n  Intercept (Control, Pre): {did_model.params['const']:.3f}")
    print(f"  Treatment effect (baseline): {did_model.params['treatment']:.3f}")
    print(f"  Time effect (Post): {did_model.params['post']:.3f}")
    print(f"  DiD Effect (Treatment*Post): {did_model.params['treat_post']:.3f}")
    print(f"    p-value: {did_model.pvalues['treat_post']:.4f}")

    if did_model.pvalues["treat_post"] < 0.05:
        if did_model.params["treat_post"] < 0:
            print(f"\n  ✓ SIGNIFICANT: Expanding states saw DECREASED signal variability")
            print(f"    (More sites = more stable signal)")
        else:
            print(f"\n  ✓ SIGNIFICANT: Expanding states saw INCREASED signal variability")
    else:
        print(f"\n  ✗ DiD effect not statistically significant")

    # Show means by group
    print("\n--- Mean Signal Std by Group ---")
    means = did_data.groupby(["treatment", "post"])["signal_std"].mean().unstack()
    means.index = ["Control (Stable)", "Treatment (Expanded)"]
    means.columns = ["Pre", "Post"]
    means["Change"] = means["Post"] - means["Pre"]
    print(means.round(3).to_string())

    # Calculate DiD manually
    did_manual = (
        means.loc["Treatment (Expanded)", "Post"] - means.loc["Treatment (Expanded)", "Pre"]
    ) - (means.loc["Control (Stable)", "Post"] - means.loc["Control (Stable)", "Pre"])
    print(f"\nDiD estimate (manual): {did_manual:.3f}")

    return {
        "model": did_model,
        "did_effect": did_model.params["treat_post"],
        "did_pvalue": did_model.pvalues["treat_post"],
        "means": means.to_dict(),
        "did_manual": did_manual,
    }


def analyze_covid_surge_moderator(data_dir: Path = Path("data")) -> dict:
    """
    Test if the SVI-signal quality relationship is stronger during COVID surges.

    Hypothesis: During surges, measurement precision matters more,
    so the infrastructure gap should have larger effects.
    """
    print("\n" + "=" * 70)
    print("NATURAL EXPERIMENT 4: COVID Surge as Moderator")
    print("=" * 70)

    # Load hospitalization data to identify surge periods
    nhsn = pd.read_parquet(data_dir / "raw" / "nhsn" / "nhsn_weekly_respiratory_20260111.parquet")
    nhsn["week"] = pd.to_datetime(nhsn["Week Ending Date"])

    # National total hospitalizations
    national = nhsn.groupby("week")["Total Patients Hospitalized with COVID-19"].sum().reset_index()
    national.columns = ["week", "covid_hosp"]

    # Define surge as weeks above 75th percentile
    surge_threshold = national["covid_hosp"].quantile(0.75)
    national["is_surge"] = national["covid_hosp"] > surge_threshold

    print(f"COVID hospitalization 75th percentile: {surge_threshold:,.0f}")
    print(f"Surge weeks: {national['is_surge'].sum()} out of {len(national)}")

    # Identify surge periods
    surge_weeks = national[national["is_surge"]]["week"].tolist()
    print(f"\nSurge period weeks: {len(surge_weeks)}")
    if len(surge_weeks) > 0:
        print(f"  From: {min(surge_weeks).date()} to {max(surge_weeks).date()}")

    # Load wastewater and merge with surge indicator
    nwss = pd.read_parquet(data_dir / "raw" / "nwss" / "nwss_metrics_20260111.parquet")

    state_map = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }
    nwss["state"] = nwss["wwtp_jurisdiction"].map(state_map)
    nwss["week"] = pd.to_datetime(nwss["date_end"])

    nwss = pd.merge(nwss, national[["week", "is_surge"]], on="week", how="left")
    nwss["is_surge"] = nwss["is_surge"].fillna(False)

    # Load SVI
    from src.analysis.health_equity_ratios import get_state_svi_rankings

    svi = get_state_svi_rankings()

    # Aggregate to state-surge level
    state_surge = (
        nwss.groupby(["state", "is_surge"])
        .agg({"percentile": ["mean", "std"], "wwtp_id": "nunique"})
        .reset_index()
    )
    state_surge.columns = ["state", "is_surge", "ww_mean", "ww_std", "n_sites"]

    state_surge = pd.merge(state_surge, svi, on="state", how="inner")

    # Calculate signal CV
    state_surge["signal_cv"] = state_surge["ww_std"] / state_surge["ww_mean"]

    # Test SVI-signal relationship separately for surge vs non-surge
    print("\n--- SVI vs Signal Quality by Period ---")

    for surge_val, label in [(True, "Surge"), (False, "Non-Surge")]:
        subset = state_surge[state_surge["is_surge"] == surge_val]
        if len(subset) < 10:
            continue

        corr, pval = stats.pearsonr(subset["svi_score"], subset["signal_cv"])
        print(f"\n{label} periods:")
        print(f"  SVI vs Signal CV: r={corr:.3f}, p={pval:.4f}")

        # OLS
        X = sm.add_constant(subset["svi_score"])
        y = subset["signal_cv"]
        model = sm.OLS(y, X).fit()
        print(f"  OLS coefficient: {model.params['svi_score']:.3f}")

    # Interaction model
    print("\n--- Interaction Model ---")
    print("Model: Signal_CV ~ SVI + Surge + SVI*Surge")

    state_surge["svi_surge"] = state_surge["svi_score"] * state_surge["is_surge"].astype(int)

    y = state_surge["signal_cv"].dropna()
    X = state_surge.loc[y.index, ["svi_score", "is_surge", "svi_surge"]]
    X["is_surge"] = X["is_surge"].astype(int)
    X = sm.add_constant(X)

    interact_model = sm.OLS(y, X).fit()

    print(
        f"\n  SVI main effect: {interact_model.params['svi_score']:.3f} (p={interact_model.pvalues['svi_score']:.4f})"
    )
    print(
        f"  Surge main effect: {interact_model.params['is_surge']:.3f} (p={interact_model.pvalues['is_surge']:.4f})"
    )
    print(
        f"  SVI*Surge interaction: {interact_model.params['svi_surge']:.3f} (p={interact_model.pvalues['svi_surge']:.4f})"
    )

    if interact_model.pvalues["svi_surge"] < 0.1:
        print(
            f"\n  → SVI effect is {'stronger' if interact_model.params['svi_surge'] > 0 else 'weaker'} during surges"
        )

    return {
        "surge_threshold": surge_threshold,
        "surge_weeks": len(surge_weeks),
        "interact_model": interact_model,
        "svi_surge_coef": interact_model.params["svi_surge"],
        "svi_surge_pval": interact_model.pvalues["svi_surge"],
    }


def run_all_natural_experiments(data_dir: Path = Path("data")) -> dict:
    """Run all natural experiment analyses."""

    print("\n" + "=" * 70)
    print("SEARCHING FOR NATURAL EXPERIMENTS")
    print("=" * 70)

    results = {}

    # 1. Site expansion analysis
    expansion = analyze_site_expansion(data_dir)
    results["expansion"] = expansion

    # 2. First sample dates
    onboarding = analyze_first_sample_dates(data_dir)
    results["onboarding"] = onboarding

    # 3. Difference-in-differences
    did = difference_in_differences(data_dir, expansion)
    results["did"] = did

    # 4. Surge moderator
    surge = analyze_covid_surge_moderator(data_dir)
    results["surge"] = surge

    # Summary
    print("\n" + "=" * 70)
    print("NATURAL EXPERIMENTS SUMMARY")
    print("=" * 70)

    did_effect = did.get("did_effect")
    did_pval = did.get("did_pvalue")
    surge_coef = surge.get("svi_surge_coef")
    surge_pval = surge.get("svi_surge_pval")

    # Format results safely
    did_effect_str = f"{did_effect:.3f}" if did_effect is not None else "N/A"
    did_pval_str = f"{did_pval:.4f}" if did_pval is not None else "N/A"
    did_sig = "Yes" if did_pval is not None and did_pval < 0.05 else "No"
    surge_str = (
        f"{surge_coef:.3f}" if surge_coef is not None and not np.isnan(surge_coef) else "N/A"
    )

    print(
        f"""
1. SITE EXPANSION
   - Expanding states: {len(expansion.get('expanders', []))}
   - Contracting states: {len(expansion.get('contractors', []))}
   - Usable for DiD: {'Yes' if len(expansion.get('expanders', [])) >= 5 else 'Limited'}

2. DIFFERENCE-IN-DIFFERENCES
   - DiD effect: {did_effect_str}
   - p-value: {did_pval_str}
   - Significant: {did_sig}

3. COVID SURGE MODERATION
   - Note: Limited data overlap between wastewater (2024+) and surge periods (mostly 2023)
   - SVI*Surge interaction: {surge_str}
"""
    )

    return results


if __name__ == "__main__":
    results = run_all_natural_experiments()
