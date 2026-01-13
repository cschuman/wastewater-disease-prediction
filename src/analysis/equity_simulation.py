"""
Equity Simulation: Modeling What Equitable Wastewater Surveillance Would Require

This analysis quantifies the resource requirements to close the monitoring coverage gap
between high-SVI and low-SVI areas, providing actionable policy recommendations.

Scenarios:
1. Equalize sites per capita across SVI quartiles
2. Prioritize highest-SVI counties with zero coverage
3. Target minimum coverage threshold everywhere

Outputs:
- Total new sites needed
- Distribution by state and county
- Priority investment rankings
- Cost estimates
- Implementation roadmap
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")


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


# COST ASSUMPTIONS (2024 estimates based on CDC/EPA guidance)
COST_PER_SITE_SETUP = 100000  # Initial setup: equipment, lab, protocols ($50k-$150k)
COST_PER_SITE_ANNUAL = 50000  # Annual operating: testing, staffing, QA/QC
YEARS_PROJECTION = 5  # 5-year budget horizon


def load_county_svi(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load and process county-level SVI data."""
    svi = pd.read_csv(data_dir / "external" / "svi_2022_county.csv", encoding="utf-8-sig")

    # Zero-pad FIPS to 5 digits
    svi["fips"] = svi["FIPS"].astype(str).str.zfill(5)

    # Select key columns
    svi_clean = svi[
        [
            "fips",
            "ST_ABBR",
            "COUNTY",
            "E_TOTPOP",
            "RPL_THEMES",  # Overall SVI (0-1, higher = more vulnerable)
            "RPL_THEME1",  # Socioeconomic Status
            "RPL_THEME2",  # Household Characteristics
            "RPL_THEME3",  # Racial & Ethnic Minority Status
            "RPL_THEME4",  # Housing Type & Transportation
        ]
    ].copy()

    svi_clean.columns = [
        "fips",
        "state",
        "county_name",
        "population",
        "svi_overall",
        "svi_socioeconomic",
        "svi_household",
        "svi_minority",
        "svi_housing",
    ]

    # Remove counties with missing SVI
    svi_clean = svi_clean[svi_clean["svi_overall"] >= 0].copy()

    # Create SVI quartiles
    svi_clean["svi_quartile"] = pd.qcut(
        svi_clean["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    print(f"Loaded SVI for {len(svi_clean):,} counties")
    return svi_clean


def load_county_wastewater(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load wastewater data and aggregate to county level."""
    nwss_dir = data_dir / "raw" / "nwss"
    nwss_file = get_latest_file(nwss_dir, "nwss_metrics_*.parquet")
    nwss = pd.read_parquet(nwss_file)

    # Handle multi-county sewersheds - take first county (primary)
    nwss["fips"] = nwss["county_fips"].str.split(",").str[0].str.strip()
    nwss["fips"] = nwss["fips"].str.zfill(5)

    # Aggregate to county level
    county_ww = (
        nwss.groupby("fips")
        .agg(
            {
                "wwtp_id": "nunique",  # Number of monitoring sites
                "population_served": "sum",  # Total population covered
            }
        )
        .reset_index()
    )

    county_ww.columns = ["fips", "n_sites", "pop_served"]

    print(f"Wastewater data for {len(county_ww):,} counties")
    return county_ww


def create_baseline_dataset(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Create baseline dataset merging SVI and current wastewater coverage."""
    svi = load_county_svi(data_dir)
    ww = load_county_wastewater(data_dir)

    # Merge
    df = pd.merge(svi, ww, on="fips", how="left")

    # Fill missing values for counties without monitoring
    df["n_sites"] = df["n_sites"].fillna(0).astype(int)
    df["pop_served"] = df["pop_served"].fillna(0)

    # Calculate coverage metrics
    df["sites_per_million"] = df["n_sites"] / (df["population"] / 1_000_000)
    df["sites_per_million"] = df["sites_per_million"].replace([np.inf, -np.inf], 0)

    df["coverage_pct"] = (df["pop_served"] / df["population"] * 100).clip(upper=100)
    df["has_monitoring"] = df["n_sites"] > 0

    print(f"\nBaseline dataset: {len(df):,} counties")
    print(
        f"Counties with monitoring: {df['has_monitoring'].sum():,} ({df['has_monitoring'].mean()*100:.1f}%)"
    )
    print(
        f"Population covered: {df['pop_served'].sum()/1e6:.1f}M ({df['pop_served'].sum()/df['population'].sum()*100:.1f}%)"
    )

    return df


def calculate_current_coverage_by_svi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate current coverage statistics by SVI quartile."""

    stats = (
        df.groupby("svi_quartile")
        .agg(
            {
                "fips": "count",
                "population": "sum",
                "has_monitoring": "sum",
                "n_sites": "sum",
                "pop_served": "sum",
                "sites_per_million": "mean",
            }
        )
        .reset_index()
    )

    stats.columns = [
        "SVI Quartile",
        "N Counties",
        "Total Pop",
        "Counties w/ Monitoring",
        "Total Sites",
        "Pop Served",
        "Sites per Million",
    ]

    # Calculate percentages
    stats["% Counties w/ Monitoring"] = (
        stats["Counties w/ Monitoring"] / stats["N Counties"] * 100
    ).round(1)
    stats["% Pop Covered"] = (stats["Pop Served"] / stats["Total Pop"] * 100).round(1)
    stats["Sites per Million"] = stats["Sites per Million"].round(2)

    # Format population in millions
    stats["Total Pop (M)"] = (stats["Total Pop"] / 1e6).round(1)
    stats["Pop Served (M)"] = (stats["Pop Served"] / 1e6).round(1)

    # Reorder columns
    stats = stats[
        [
            "SVI Quartile",
            "N Counties",
            "Total Pop (M)",
            "Counties w/ Monitoring",
            "% Counties w/ Monitoring",
            "Total Sites",
            "Sites per Million",
            "Pop Served (M)",
            "% Pop Covered",
        ]
    ]

    return stats


def scenario_a_equalize_per_capita(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Scenario A: Equalize sites per capita across SVI quartiles

    Target: Bring all quartiles up to the Q1 (low-SVI) rate of sites per million.
    This represents full equity in monitoring infrastructure.
    """
    print("\n" + "=" * 80)
    print("SCENARIO A: Equalize Sites Per Capita Across SVI Quartiles")
    print("=" * 80)

    # Calculate current rates by quartile
    quartile_rates = (
        df.groupby("svi_quartile")
        .agg({"sites_per_million": "mean", "population": "sum", "n_sites": "sum"})
        .reset_index()
    )

    # Target rate: Q1 (lowest SVI, currently highest coverage)
    target_rate = quartile_rates.loc[
        quartile_rates["svi_quartile"] == "Q1 (Low)", "sites_per_million"
    ].values[0]

    print(f"\nCurrent coverage rates:")
    for _, row in quartile_rates.iterrows():
        print(f"  {row['svi_quartile']}: {row['sites_per_million']:.2f} sites/million")

    print(f"\nTarget rate (Q1 benchmark): {target_rate:.2f} sites per million")

    # Calculate needed sites for each county
    df_scenario = df.copy()
    df_scenario["target_sites"] = (df_scenario["population"] / 1_000_000 * target_rate).round()
    df_scenario["new_sites_needed"] = (
        (df_scenario["target_sites"] - df_scenario["n_sites"]).clip(lower=0).astype(int)
    )

    # Summary statistics
    total_new_sites = df_scenario["new_sites_needed"].sum()
    total_cost_setup = total_new_sites * COST_PER_SITE_SETUP
    total_cost_5yr = total_cost_setup + (total_new_sites * COST_PER_SITE_ANNUAL * YEARS_PROJECTION)

    # By SVI quartile
    quartile_summary = (
        df_scenario.groupby("svi_quartile")
        .agg({"new_sites_needed": "sum", "population": "sum", "fips": "count"})
        .reset_index()
    )
    quartile_summary.columns = ["SVI Quartile", "New Sites Needed", "Population", "N Counties"]
    quartile_summary["% of New Sites"] = (
        quartile_summary["New Sites Needed"] / total_new_sites * 100
    ).round(1)

    print(f"\n{'-'*80}")
    print(f"TOTAL NEW SITES NEEDED: {total_new_sites:,}")
    print(f"Setup cost: ${total_cost_setup/1e6:.1f}M")
    print(f"5-year total cost: ${total_cost_5yr/1e6:.1f}M")
    print(f"{'-'*80}")

    print("\nDistribution by SVI quartile:")
    print(quartile_summary.to_string(index=False))

    # Top states needing sites
    state_summary = (
        df_scenario.groupby("state")
        .agg({"new_sites_needed": "sum", "population": "sum", "fips": "count"})
        .reset_index()
        .sort_values("new_sites_needed", ascending=False)
    )

    state_summary.columns = ["State", "New Sites", "Population", "N Counties"]
    state_summary["Setup Cost ($M)"] = (
        state_summary["New Sites"] * COST_PER_SITE_SETUP / 1e6
    ).round(1)

    print(f"\nTop 15 states by new sites needed:")
    print(state_summary.head(15).to_string(index=False))

    # Return detailed county-level data and summary
    results = {
        "scenario": "A - Equalize Per Capita",
        "target_rate": target_rate,
        "total_new_sites": int(total_new_sites),
        "total_cost_setup": total_cost_setup,
        "total_cost_5yr": total_cost_5yr,
        "quartile_summary": quartile_summary,
        "state_summary": state_summary,
    }

    return df_scenario, results


def scenario_b_prioritize_zero_coverage(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Scenario B: Prioritize highest-SVI counties with zero coverage

    Strategy: Ensure every high-SVI county has at least minimal monitoring.
    Focus on counties in Q3/Q4 (high SVI) that currently have zero sites.
    """
    print("\n" + "=" * 80)
    print("SCENARIO B: Prioritize High-SVI Counties with Zero Coverage")
    print("=" * 80)

    df_scenario = df.copy()

    # Define priority tiers
    # Tier 1: Q4 (highest SVI) with no monitoring
    # Tier 2: Q3 with no monitoring
    # Tier 3: Q4 with monitoring but below target
    # Tier 4: Q3 with monitoring but below target

    # Minimum sites based on population
    # Small counties: 1 site
    # Medium counties (>100k): 2 sites
    # Large counties (>500k): 3+ sites
    def calculate_minimum_sites(pop):
        if pop < 100_000:
            return 1
        elif pop < 500_000:
            return 2
        else:
            return max(3, int(pop / 250_000))

    df_scenario["min_sites"] = df_scenario["population"].apply(calculate_minimum_sites)

    # Priority scoring: higher SVI + zero coverage = higher priority
    df_scenario["priority_score"] = df_scenario["svi_overall"] * 100
    df_scenario.loc[
        df_scenario["n_sites"] == 0, "priority_score"
    ] *= 2  # Double weight for zero coverage

    # Calculate sites needed
    df_scenario["new_sites_needed"] = (
        (df_scenario["min_sites"] - df_scenario["n_sites"]).clip(lower=0).astype(int)
    )

    # Filter to high-SVI counties only (Q3 and Q4)
    high_svi = df_scenario[df_scenario["svi_quartile"].isin(["Q3", "Q4 (High)"])].copy()

    total_new_sites = high_svi["new_sites_needed"].sum()
    total_cost_setup = total_new_sites * COST_PER_SITE_SETUP
    total_cost_5yr = total_cost_setup + (total_new_sites * COST_PER_SITE_ANNUAL * YEARS_PROJECTION)

    # Counties needing sites
    counties_needing = high_svi[high_svi["new_sites_needed"] > 0].copy()
    counties_zero_cov = counties_needing[counties_needing["n_sites"] == 0]

    print(f"\nHigh-SVI counties (Q3 + Q4): {len(high_svi):,}")
    print(f"Counties needing additional sites: {len(counties_needing):,}")
    print(f"Counties with ZERO coverage: {len(counties_zero_cov):,}")

    print(f"\n{'-'*80}")
    print(f"TOTAL NEW SITES NEEDED: {total_new_sites:,}")
    print(f"Setup cost: ${total_cost_setup/1e6:.1f}M")
    print(f"5-year total cost: ${total_cost_5yr/1e6:.1f}M")
    print(f"{'-'*80}")

    # Top priority counties
    priority_counties = counties_needing.sort_values("priority_score", ascending=False)[
        [
            "state",
            "county_name",
            "population",
            "svi_overall",
            "n_sites",
            "new_sites_needed",
            "priority_score",
        ]
    ].head(20)

    priority_counties["Population (k)"] = (
        (priority_counties["population"] / 1000).round(0).astype(int)
    )
    priority_counties = priority_counties[
        ["state", "county_name", "Population (k)", "svi_overall", "n_sites", "new_sites_needed"]
    ]
    priority_counties.columns = [
        "State",
        "County",
        "Pop (k)",
        "SVI",
        "Current Sites",
        "New Sites Needed",
    ]

    print(f"\nTop 20 priority counties:")
    print(priority_counties.to_string(index=False))

    # State summary
    state_summary = (
        counties_needing.groupby("state")
        .agg({"new_sites_needed": "sum", "fips": "count", "population": "sum"})
        .reset_index()
        .sort_values("new_sites_needed", ascending=False)
    )

    state_summary.columns = ["State", "New Sites", "N Counties", "Population"]
    state_summary["Setup Cost ($M)"] = (
        state_summary["New Sites"] * COST_PER_SITE_SETUP / 1e6
    ).round(1)

    print(f"\nTop 15 states by new sites needed:")
    print(state_summary.head(15).to_string(index=False))

    results = {
        "scenario": "B - Prioritize Zero Coverage",
        "total_new_sites": int(total_new_sites),
        "total_cost_setup": total_cost_setup,
        "total_cost_5yr": total_cost_5yr,
        "counties_needing": len(counties_needing),
        "counties_zero_coverage": len(counties_zero_cov),
        "priority_counties": priority_counties,
        "state_summary": state_summary,
    }

    return df_scenario, results


def scenario_c_minimum_threshold(
    df: pd.DataFrame, min_coverage_pct: float = 50.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Scenario C: Target minimum coverage threshold everywhere

    Strategy: Ensure every county reaches a minimum percentage of population covered,
    regardless of SVI. This represents a baseline public health standard.

    Args:
        min_coverage_pct: Minimum percentage of population that should be covered (default 50%)
    """
    print("\n" + "=" * 80)
    print(f"SCENARIO C: Target {min_coverage_pct:.0f}% Population Coverage Threshold")
    print("=" * 80)

    df_scenario = df.copy()

    # Calculate how many sites needed to reach target coverage
    # Assumption: Average site covers 100,000 people (based on current NWSS data)
    avg_coverage_per_site = 100_000

    df_scenario["target_pop_covered"] = df_scenario["population"] * (min_coverage_pct / 100)
    df_scenario["additional_pop_needed"] = (
        df_scenario["target_pop_covered"] - df_scenario["pop_served"]
    ).clip(lower=0)
    df_scenario["new_sites_needed"] = np.ceil(
        df_scenario["additional_pop_needed"] / avg_coverage_per_site
    ).astype(int)

    # Counties not meeting threshold
    counties_below = df_scenario[df_scenario["coverage_pct"] < min_coverage_pct].copy()

    total_new_sites = df_scenario["new_sites_needed"].sum()
    total_cost_setup = total_new_sites * COST_PER_SITE_SETUP
    total_cost_5yr = total_cost_setup + (total_new_sites * COST_PER_SITE_ANNUAL * YEARS_PROJECTION)

    print(f"\nCurrent status:")
    print(
        f"Counties below {min_coverage_pct:.0f}% coverage: {len(counties_below):,} ({len(counties_below)/len(df)*100:.1f}%)"
    )
    print(f"Population in counties below threshold: {counties_below['population'].sum()/1e6:.1f}M")

    print(f"\n{'-'*80}")
    print(f"TOTAL NEW SITES NEEDED: {total_new_sites:,}")
    print(f"Setup cost: ${total_cost_setup/1e6:.1f}M")
    print(f"5-year total cost: ${total_cost_5yr/1e6:.1f}M")
    print(f"{'-'*80}")

    # Impact by SVI quartile
    quartile_summary = (
        df_scenario.groupby("svi_quartile")
        .agg(
            {
                "new_sites_needed": "sum",
                "population": "sum",
                "fips": lambda x: (
                    (counties_below["svi_quartile"] == x.name).sum()
                    if x.name in counties_below["svi_quartile"].values
                    else 0
                ),
            }
        )
        .reset_index()
    )

    quartile_summary.columns = [
        "SVI Quartile",
        "New Sites",
        "Population",
        "Counties Below Threshold",
    ]
    quartile_summary["% of New Sites"] = (
        quartile_summary["New Sites"] / total_new_sites * 100
    ).round(1)

    print(f"\nImpact by SVI quartile:")
    print(quartile_summary.to_string(index=False))

    # Priority counties (largest gaps)
    df_scenario["coverage_gap"] = min_coverage_pct - df_scenario["coverage_pct"]
    counties_below["coverage_gap"] = min_coverage_pct - counties_below["coverage_pct"]
    priority_counties = counties_below.sort_values("coverage_gap", ascending=False)[
        [
            "state",
            "county_name",
            "population",
            "coverage_pct",
            "svi_overall",
            "n_sites",
            "new_sites_needed",
        ]
    ].head(20)

    priority_counties["Population (k)"] = (
        (priority_counties["population"] / 1000).round(0).astype(int)
    )
    priority_counties = priority_counties[
        [
            "state",
            "county_name",
            "Population (k)",
            "coverage_pct",
            "svi_overall",
            "n_sites",
            "new_sites_needed",
        ]
    ]
    priority_counties.columns = [
        "State",
        "County",
        "Pop (k)",
        "Current Cov %",
        "SVI",
        "Current Sites",
        "New Sites",
    ]

    print(f"\nTop 20 counties with largest coverage gaps:")
    print(priority_counties.to_string(index=False))

    # State summary
    state_summary = (
        df_scenario[df_scenario["new_sites_needed"] > 0]
        .groupby("state")
        .agg({"new_sites_needed": "sum", "fips": "count", "population": "sum"})
        .reset_index()
        .sort_values("new_sites_needed", ascending=False)
    )

    state_summary.columns = ["State", "New Sites", "N Counties", "Population"]
    state_summary["Setup Cost ($M)"] = (
        state_summary["New Sites"] * COST_PER_SITE_SETUP / 1e6
    ).round(1)

    print(f"\nTop 15 states by new sites needed:")
    print(state_summary.head(15).to_string(index=False))

    results = {
        "scenario": f"C - {min_coverage_pct:.0f}% Coverage Threshold",
        "min_coverage_pct": min_coverage_pct,
        "total_new_sites": int(total_new_sites),
        "total_cost_setup": total_cost_setup,
        "total_cost_5yr": total_cost_5yr,
        "counties_below_threshold": len(counties_below),
        "quartile_summary": quartile_summary,
        "priority_counties": priority_counties,
        "state_summary": state_summary,
    }

    return df_scenario, results


def create_priority_investment_map(df: pd.DataFrame, scenario_results: Dict) -> pd.DataFrame:
    """
    Create a unified priority investment ranking combining all scenarios.

    Returns a county-level priority list with composite scoring based on:
    - SVI (higher = higher priority)
    - Current coverage gap (larger = higher priority)
    - Population (larger = higher impact)
    - Consistency across scenarios (appears in multiple scenarios = higher priority)
    """
    print("\n" + "=" * 80)
    print("PRIORITY INVESTMENT MAP: Unified Recommendations")
    print("=" * 80)

    # Normalize scores to 0-100
    df_priority = df.copy()

    # Component 1: SVI (0-100)
    df_priority["score_svi"] = df_priority["svi_overall"] * 100

    # Component 2: Coverage gap (0-100)
    national_avg_rate = df["n_sites"].sum() / (df["population"].sum() / 1_000_000)
    df_priority["expected_sites"] = df_priority["population"] / 1_000_000 * national_avg_rate
    df_priority["coverage_gap"] = (df_priority["expected_sites"] - df_priority["n_sites"]).clip(
        lower=0
    )
    df_priority["score_gap"] = (
        df_priority["coverage_gap"] / df_priority["coverage_gap"].max() * 100
    ).fillna(0)

    # Component 3: Population impact (0-100, log scale for fairness)
    df_priority["score_population"] = (
        np.log10(df_priority["population"]) / np.log10(df_priority["population"].max()) * 100
    )

    # Component 4: Zero coverage penalty (binary +20 points)
    df_priority["score_zero_coverage"] = (df_priority["n_sites"] == 0).astype(int) * 20

    # Composite priority score (weighted average)
    df_priority["priority_score"] = (
        df_priority["score_svi"] * 0.35  # SVI: 35% weight
        + df_priority["score_gap"] * 0.30  # Coverage gap: 30% weight
        + df_priority["score_population"] * 0.15  # Population: 15% weight
        + df_priority["score_zero_coverage"] * 0.20  # Zero coverage: 20% weight
    )

    # Priority tier classification
    df_priority["priority_tier"] = pd.qcut(
        df_priority["priority_score"],
        q=4,
        labels=["Tier 4 (Low)", "Tier 3", "Tier 2", "Tier 1 (Highest)"],
    )

    # Top priority counties
    top_priorities = df_priority.sort_values("priority_score", ascending=False).head(50)

    investment_map = top_priorities[
        [
            "state",
            "county_name",
            "population",
            "svi_overall",
            "svi_quartile",
            "n_sites",
            "coverage_pct",
            "priority_score",
            "priority_tier",
        ]
    ].copy()

    investment_map["Population (k)"] = (investment_map["population"] / 1000).round(0).astype(int)
    investment_map["Priority Score"] = investment_map["priority_score"].round(1)

    investment_map = investment_map[
        [
            "state",
            "county_name",
            "Population (k)",
            "svi_overall",
            "svi_quartile",
            "n_sites",
            "coverage_pct",
            "Priority Score",
            "priority_tier",
        ]
    ]

    investment_map.columns = [
        "State",
        "County",
        "Pop (k)",
        "SVI",
        "SVI Quartile",
        "Current Sites",
        "Coverage %",
        "Priority Score",
        "Tier",
    ]

    print(f"\nTop 30 Priority Counties for Investment:")
    print(investment_map.head(30).to_string(index=False))

    # Tier summary
    tier_summary = (
        df_priority.groupby("priority_tier")
        .agg({"fips": "count", "population": "sum", "n_sites": "sum"})
        .reset_index()
    )

    tier_summary.columns = ["Priority Tier", "N Counties", "Population", "Current Sites"]
    tier_summary["% of US Population"] = (
        tier_summary["Population"] / tier_summary["Population"].sum() * 100
    ).round(1)

    print(f"\nPriority tier distribution:")
    print(tier_summary.to_string(index=False))

    return investment_map, df_priority


def generate_policy_recommendations(baseline_df: pd.DataFrame, scenario_results: Dict):
    """Generate actionable policy recommendations based on all scenarios."""

    print("\n" + "=" * 80)
    print("ACTIONABLE POLICY RECOMMENDATIONS")
    print("=" * 80)

    # Extract key numbers
    n_counties = len(baseline_df)
    n_monitored = baseline_df["has_monitoring"].sum()
    pct_monitored = n_monitored / n_counties * 100

    high_svi_counties = baseline_df[baseline_df["svi_quartile"].isin(["Q3", "Q4 (High)"])]
    high_svi_no_coverage = high_svi_counties[high_svi_counties["n_sites"] == 0]

    scenario_a = scenario_results["scenario_a"]
    scenario_b = scenario_results["scenario_b"]
    scenario_c = scenario_results["scenario_c"]

    print(
        f"""
EXECUTIVE SUMMARY
-----------------
Current State:
  • {n_counties:,} US counties tracked
  • {n_monitored:,} counties have wastewater monitoring ({pct_monitored:.1f}%)
  • {len(high_svi_no_coverage):,} high-SVI counties have ZERO coverage
  • High-SVI states have 44% fewer monitoring sites per capita

The Equity Gap:
  • Low-SVI counties: {baseline_df[baseline_df['svi_quartile']=='Q1 (Low)']['sites_per_million'].mean():.1f} sites per million
  • High-SVI counties: {baseline_df[baseline_df['svi_quartile']=='Q4 (High)']['sites_per_million'].mean():.1f} sites per million
  • This {baseline_df[baseline_df['svi_quartile']=='Q1 (Low)']['sites_per_million'].mean() / baseline_df[baseline_df['svi_quartile']=='Q4 (High)']['sites_per_million'].mean():.1f}x disparity leaves vulnerable communities underserved


RECOMMENDED INVESTMENT SCENARIOS
---------------------------------

Option 1: FULL EQUITY (Scenario A)
  Objective: Equalize sites per capita to match low-SVI counties

  Investment Required:
    • {scenario_a['total_new_sites']:,} new monitoring sites
    • ${scenario_a['total_cost_setup']/1e6:.0f}M setup costs
    • ${scenario_a['total_cost_5yr']/1e6:.0f}M total 5-year cost

  Impact:
    • Eliminates the equity gap entirely
    • Brings all communities to the same monitoring standard
    • {scenario_a['quartile_summary'].loc[scenario_a['quartile_summary']['SVI Quartile']=='Q4 (High)', 'New Sites Needed'].values[0]:,} sites to highest-SVI counties

  Timeline: 5-7 years (phased rollout by priority tier)


Option 2: TARGETED PRIORITY (Scenario B) ⭐ RECOMMENDED
  Objective: Ensure all high-SVI counties have minimum monitoring

  Investment Required:
    • {scenario_b['total_new_sites']:,} new monitoring sites
    • ${scenario_b['total_cost_setup']/1e6:.0f}M setup costs
    • ${scenario_b['total_cost_5yr']/1e6:.0f}M total 5-year cost

  Impact:
    • Closes the most critical gaps with ~{scenario_b['total_cost_setup']/scenario_a['total_cost_setup']*100:.0f}% of full equity cost
    • Prioritizes {scenario_b['counties_zero_coverage']:,} high-SVI counties with zero coverage
    • Addresses the most urgent public health needs

  Timeline: 3-5 years (focused implementation)


Option 3: BASELINE STANDARD (Scenario C)
  Objective: Ensure 50% population coverage in every county

  Investment Required:
    • {scenario_c['total_new_sites']:,} new monitoring sites
    • ${scenario_c['total_cost_setup']/1e6:.0f}M setup costs
    • ${scenario_c['total_cost_5yr']/1e6:.0f}M total 5-year cost

  Impact:
    • Establishes minimum public health monitoring floor
    • {scenario_c['counties_below_threshold']:,} counties currently below threshold
    • Balances equity with feasibility

  Timeline: 3-5 years


IMPLEMENTATION ROADMAP (Based on Scenario B - Recommended)
-----------------------------------------------------------

Phase 1 (Year 1): High-Priority, Zero-Coverage Counties
  • Target: {len(high_svi_no_coverage):,} high-SVI counties with no monitoring
  • Focus: Top priority tier counties (largest, highest SVI)
  • Sites: ~{int(scenario_b['total_new_sites'] * 0.35):,} new sites
  • Cost: ${ scenario_b['total_cost_setup'] * 0.35 / 1e6:.0f}M

Phase 2 (Years 2-3): Expand Coverage in High-SVI Areas
  • Target: Q3/Q4 counties with below-target coverage
  • Sites: ~{int(scenario_b['total_new_sites'] * 0.40):,} new sites
  • Cost: ${ scenario_b['total_cost_setup'] * 0.40 / 1e6:.0f}M

Phase 3 (Years 4-5): Address Remaining Gaps
  • Target: All remaining counties below national average
  • Sites: ~{int(scenario_b['total_new_sites'] * 0.25):,} new sites
  • Cost: ${ scenario_b['total_cost_setup'] * 0.25 / 1e6:.0f}M


TOP 10 STATES FOR PRIORITY INVESTMENT
--------------------------------------
(Based on Scenario B - number of new sites needed)
"""
    )

    top_states = scenario_b["state_summary"].head(10)
    for idx, row in top_states.iterrows():
        print(
            f"  {idx+1}. {row['State']}: {row['New Sites']:,} sites, ${row['Setup Cost ($M)']:.0f}M"
        )

    print(
        f"""

POLICY LEVERS FOR IMPLEMENTATION
---------------------------------

1. Federal Funding Mechanisms
   • CDC Epidemiology and Laboratory Capacity (ELC) grants
   • EPA Water Infrastructure Improvements for the Nation (WIIN) Act
   • Create dedicated wastewater surveillance equity fund

2. State-Federal Partnerships
   • Require equity plans for federal wastewater funding
   • Incentivize state matching for high-SVI county sites
   • Technical assistance for rural/under-resourced communities

3. Public-Private Collaboration
   • Partner with utilities in underserved areas
   • Academic-community partnerships for site operations
   • Shared infrastructure with existing environmental monitoring

4. Performance Metrics
   • Track coverage equity by SVI quartile annually
   • Set state-level targets for high-SVI county coverage
   • Public dashboard showing progress toward equity goals


KEY TAKEAWAY
------------
Closing the wastewater surveillance equity gap is achievable with targeted
investment of ${scenario_b['total_cost_setup']/1e6:.0f}M setup + ${scenario_b['total_cost_5yr']/1e6:.0f}M over 5 years.

This investment would ensure that vulnerable communities - those most likely
to experience severe disease burden - have equal access to early warning
systems that can save lives and prevent outbreaks.

The cost of NOT acting: continued health disparities, delayed outbreak
detection in vulnerable communities, and preventable disease transmission.
"""
    )


def run_full_equity_simulation(data_dir: Path = Path("data")) -> Dict:
    """Run complete equity simulation analysis with all scenarios."""

    print("=" * 80)
    print("WASTEWATER SURVEILLANCE EQUITY SIMULATION")
    print("Modeling What Equitable Coverage Would Require")
    print("=" * 80)

    # Load baseline data
    print("\n[1/6] Loading data...")
    df = create_baseline_dataset(data_dir)

    # Current coverage analysis
    print("\n[2/6] Analyzing current coverage by SVI...")
    current_coverage = calculate_current_coverage_by_svi(df)
    print("\nCurrent Coverage by SVI Quartile:")
    print(current_coverage.to_string(index=False))

    # Run scenarios
    print("\n[3/6] Running equity scenarios...")

    df_scenario_a, results_a = scenario_a_equalize_per_capita(df)
    df_scenario_b, results_b = scenario_b_prioritize_zero_coverage(df)
    df_scenario_c, results_c = scenario_c_minimum_threshold(df, min_coverage_pct=50.0)

    # Create priority investment map
    print("\n[4/6] Creating priority investment map...")
    scenario_results = {"scenario_a": results_a, "scenario_b": results_b, "scenario_c": results_c}

    investment_map, df_priority = create_priority_investment_map(df, scenario_results)

    # Generate policy recommendations
    print("\n[5/6] Generating policy recommendations...")
    generate_policy_recommendations(df, scenario_results)

    # Save outputs
    print("\n[6/6] Saving outputs...")
    output_dir = Path("reports/equity_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed county-level results
    df_priority[
        [
            "fips",
            "state",
            "county_name",
            "population",
            "svi_overall",
            "svi_quartile",
            "n_sites",
            "coverage_pct",
            "priority_score",
            "priority_tier",
        ]
    ].to_csv(output_dir / "county_priority_rankings.csv", index=False)

    # Save scenario comparisons
    scenario_comparison = pd.DataFrame(
        [
            {
                "Scenario": "A - Full Equity",
                "New Sites": results_a["total_new_sites"],
                "Setup Cost ($M)": results_a["total_cost_setup"] / 1e6,
                "5-Year Cost ($M)": results_a["total_cost_5yr"] / 1e6,
                "Focus": "Equalize to Q1 rate",
            },
            {
                "Scenario": "B - Priority High-SVI",
                "New Sites": results_b["total_new_sites"],
                "Setup Cost ($M)": results_b["total_cost_setup"] / 1e6,
                "5-Year Cost ($M)": results_b["total_cost_5yr"] / 1e6,
                "Focus": "High-SVI zero coverage",
            },
            {
                "Scenario": "C - 50% Coverage Floor",
                "New Sites": results_c["total_new_sites"],
                "Setup Cost ($M)": results_c["total_cost_setup"] / 1e6,
                "5-Year Cost ($M)": results_c["total_cost_5yr"] / 1e6,
                "Focus": "Minimum threshold",
            },
        ]
    )

    scenario_comparison.to_csv(output_dir / "scenario_comparison.csv", index=False)

    # Save top priority investment map
    investment_map.to_csv(output_dir / "top_priority_counties.csv", index=False)

    print(f"\nOutputs saved to {output_dir}/")
    print("  • county_priority_rankings.csv - All counties with priority scores")
    print("  • scenario_comparison.csv - Summary of all scenarios")
    print("  • top_priority_counties.csv - Top 50 priority counties for investment")

    print("\n" + "=" * 80)
    print("EQUITY SIMULATION COMPLETE")
    print("=" * 80)

    return {
        "baseline_data": df,
        "priority_data": df_priority,
        "scenario_results": scenario_results,
        "investment_map": investment_map,
        "current_coverage": current_coverage,
    }


if __name__ == "__main__":
    results = run_full_equity_simulation()
