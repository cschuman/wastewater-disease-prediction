#!/usr/bin/env python3
"""
Convert CSV data files to optimized JSON for SvelteKit static site.
Run during build: python scripts/build-data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
WEB_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = WEB_DIR.parent

# Source data paths
EQUITY_SIM_DIR = PROJECT_ROOT / "reports" / "equity_simulation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output path
OUTPUT_DIR = WEB_DIR / "static" / "data"


def build_counties_json():
    """Merge county data and export as JSON."""
    print("Building counties.json...")

    rankings_path = EQUITY_SIM_DIR / "county_priority_rankings.csv"
    svi_rucc_path = PROCESSED_DIR / "county_svi_rucc_monitoring.csv"

    if not rankings_path.exists():
        print(f"  ERROR: {rankings_path} not found")
        return False

    rankings = pd.read_csv(rankings_path, dtype={'fips': str})

    # Try to merge with SVI/RUCC data if available
    if svi_rucc_path.exists():
        svi_rucc = pd.read_csv(svi_rucc_path, dtype={'fips': str})

        # Select columns to merge
        merge_cols = ['fips']
        optional_cols = [
            'svi_socioeconomic', 'svi_household', 'svi_minority', 'svi_housing',
            'rucc_2023', 'is_metro', 'is_rural', 'metro_size', 'urbanization'
        ]
        for col in optional_cols:
            if col in svi_rucc.columns and col not in rankings.columns:
                merge_cols.append(col)

        if len(merge_cols) > 1:
            merged = rankings.merge(svi_rucc[merge_cols], on='fips', how='left')
        else:
            merged = rankings
    else:
        print(f"  WARNING: {svi_rucc_path} not found, using rankings only")
        merged = rankings

    # Ensure fips is zero-padded to 5 digits
    merged['fips'] = merged['fips'].str.zfill(5)

    # Round floats to reduce file size
    float_cols = merged.select_dtypes(include=['float64']).columns
    merged[float_cols] = merged[float_cols].round(4)

    # Convert to records
    records = merged.to_dict(orient='records')

    # Write JSON
    output_path = OUTPUT_DIR / "counties.json"
    with open(output_path, 'w') as f:
        json.dump(records, f, separators=(',', ':'))

    print(f"  Created {output_path} ({len(records)} counties, {output_path.stat().st_size / 1024:.1f} KB)")
    return True


def build_scenarios_json():
    """Export investment scenarios as JSON."""
    print("Building scenarios.json...")

    scenarios_path = EQUITY_SIM_DIR / "scenario_comparison.csv"

    # Create scenarios with proper field names for the frontend
    if scenarios_path.exists():
        df = pd.read_csv(scenarios_path)
        scenarios = []
        for _, row in df.iterrows():
            scenario_name = row.get('Scenario', '')
            scenarios.append({
                "id": scenario_name[0].lower() if scenario_name else '',
                "name": scenario_name.split(' - ')[1] if ' - ' in scenario_name else scenario_name,
                "description": row.get('Focus', ''),
                "new_sites": int(row.get('New Sites', 0)),
                "setup_cost_millions": float(row.get('Setup Cost ($M)', 0)),
                "five_year_cost_millions": float(row.get('5-Year Cost ($M)', 0)),
                "timeline": "3-5 years",
                "recommended": 'Priority' in scenario_name or 'B' in scenario_name.split(' - ')[0]
            })
    else:
        print(f"  WARNING: {scenarios_path} not found, creating default scenarios")
        scenarios = [
            {
                "id": "a",
                "name": "Full Equity",
                "description": "Equalize sites per capita to match low-SVI counties",
                "new_sites": 799,
                "setup_cost_millions": 79.9,
                "five_year_cost_millions": 279.7,
                "timeline": "5-7 years",
                "recommended": False
            },
            {
                "id": "b",
                "name": "Targeted Priority",
                "description": "Ensure all high-SVI counties have minimum monitoring",
                "new_sites": 1720,
                "setup_cost_millions": 172.0,
                "five_year_cost_millions": 602.0,
                "timeline": "3-5 years",
                "recommended": True
            },
            {
                "id": "c",
                "name": "Baseline Standard",
                "description": "Ensure 50% population coverage in every county",
                "new_sites": 2845,
                "setup_cost_millions": 284.5,
                "five_year_cost_millions": 995.8,
                "timeline": "3-5 years",
                "recommended": False
            }
        ]

    output_path = OUTPUT_DIR / "scenarios.json"
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, separators=(',', ':'))

    print(f"  Created {output_path} ({len(scenarios)} scenarios)")
    return True


def build_top_priority_json():
    """Export top priority counties as JSON."""
    print("Building top-priority.json...")

    top_path = EQUITY_SIM_DIR / "top_priority_counties.csv"

    if not top_path.exists():
        print(f"  WARNING: {top_path} not found, skipping")
        return False

    df = pd.read_csv(top_path)

    # Rename columns to match our schema
    column_map = {
        'State': 'state',
        'County': 'county_name',
        'Pop (k)': 'population_k',
        'SVI': 'svi_overall',
        'SVI Quartile': 'svi_quartile',
        'Current Sites': 'n_sites',
        'Coverage %': 'coverage_pct',
        'Priority Score': 'priority_score',
        'Tier': 'priority_tier'
    }
    df = df.rename(columns=column_map)

    # Convert population from thousands to actual
    if 'population_k' in df.columns:
        df['population'] = (df['population_k'] * 1000).astype(int)
        df = df.drop(columns=['population_k'])

    # Round floats
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(4)

    records = df.to_dict(orient='records')

    output_path = OUTPUT_DIR / "top-priority.json"
    with open(output_path, 'w') as f:
        json.dump(records, f, separators=(',', ':'))

    print(f"  Created {output_path} ({len(records)} counties)")
    return True


def build_state_aggregates_json():
    """Create state-level summary statistics."""
    print("Building state-aggregates.json...")

    rankings_path = EQUITY_SIM_DIR / "county_priority_rankings.csv"

    if not rankings_path.exists():
        print(f"  ERROR: {rankings_path} not found")
        return False

    rankings = pd.read_csv(rankings_path, dtype={'fips': str})

    state_stats = rankings.groupby('state').agg({
        'population': 'sum',
        'n_sites': 'sum',
        'svi_overall': 'mean',
        'priority_score': 'mean',
        'fips': 'count'
    }).rename(columns={'fips': 'county_count'}).reset_index()

    state_stats['sites_per_100k'] = (
        state_stats['n_sites'] / state_stats['population'] * 100000
    ).round(2)

    # Round other floats
    state_stats['svi_overall'] = state_stats['svi_overall'].round(4)
    state_stats['priority_score'] = state_stats['priority_score'].round(2)

    records = state_stats.to_dict(orient='records')

    output_path = OUTPUT_DIR / "state-aggregates.json"
    with open(output_path, 'w') as f:
        json.dump(records, f, separators=(',', ':'))

    print(f"  Created {output_path} ({len(records)} states)")
    return True


def build_svi_quartile_stats_json():
    """Aggregate statistics by SVI quartile."""
    print("Building svi-quartile-stats.json...")

    rankings_path = EQUITY_SIM_DIR / "county_priority_rankings.csv"

    if not rankings_path.exists():
        print(f"  ERROR: {rankings_path} not found")
        return False

    rankings = pd.read_csv(rankings_path, dtype={'fips': str})

    # Standardize quartile names
    rankings['svi_quartile_clean'] = rankings['svi_quartile'].str.replace(r'\s*\(.*\)', '', regex=True)

    quartile_stats = rankings.groupby('svi_quartile').agg({
        'population': 'sum',
        'n_sites': 'sum',
        'coverage_pct': 'mean',
        'fips': 'count'
    }).rename(columns={'fips': 'county_count'}).reset_index()

    quartile_stats['sites_per_million'] = (
        quartile_stats['n_sites'] / quartile_stats['population'] * 1_000_000
    ).round(2)

    quartile_stats['coverage_pct'] = quartile_stats['coverage_pct'].round(2)

    records = quartile_stats.to_dict(orient='records')

    output_path = OUTPUT_DIR / "svi-quartile-stats.json"
    with open(output_path, 'w') as f:
        json.dump(records, f, separators=(',', ':'))

    print(f"  Created {output_path} ({len(records)} quartiles)")
    return True


def build_summary_stats_json():
    """Create summary statistics for dashboard."""
    print("Building summary-stats.json...")

    rankings_path = EQUITY_SIM_DIR / "county_priority_rankings.csv"

    if not rankings_path.exists():
        print(f"  ERROR: {rankings_path} not found")
        return False

    rankings = pd.read_csv(rankings_path, dtype={'fips': str})

    # Calculate key metrics
    total_counties = len(rankings)
    total_population = int(rankings['population'].sum())
    total_sites = int(rankings['n_sites'].sum())

    # Counties with monitoring
    counties_with_monitoring = int((rankings['n_sites'] > 0).sum())

    # High-SVI stats
    high_svi = rankings[rankings['svi_quartile'].str.contains('Q4|High', case=False, na=False)]
    high_svi_zero_coverage = int((high_svi['n_sites'] == 0).sum())

    # Calculate disparity
    q1 = rankings[rankings['svi_quartile'].str.contains('Q1', case=False, na=False)]
    q4 = rankings[rankings['svi_quartile'].str.contains('Q4|High', case=False, na=False)]

    q1_sites_per_million = (q1['n_sites'].sum() / q1['population'].sum() * 1_000_000)
    q4_sites_per_million = (q4['n_sites'].sum() / q4['population'].sum() * 1_000_000)
    disparity_pct = ((q1_sites_per_million - q4_sites_per_million) / q1_sites_per_million * 100)

    summary = {
        "total_counties": total_counties,
        "total_population": total_population,
        "total_sites": total_sites,
        "counties_with_monitoring": counties_with_monitoring,
        "coverage_pct": round(counties_with_monitoring / total_counties * 100, 1),
        "high_svi_zero_coverage": high_svi_zero_coverage,
        "q1_sites_per_million": round(q1_sites_per_million, 2),
        "q4_sites_per_million": round(q4_sites_per_million, 2),
        "disparity_pct": round(disparity_pct, 1),
        "analysis_date": "2026-01-11"
    }

    output_path = OUTPUT_DIR / "summary-stats.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Created {output_path}")
    return True


def main():
    """Run all data build functions."""
    print("=" * 60)
    print("Building static JSON data for dashboard")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all data files
    results = []
    results.append(("counties", build_counties_json()))
    results.append(("scenarios", build_scenarios_json()))
    results.append(("top-priority", build_top_priority_json()))
    results.append(("state-aggregates", build_state_aggregates_json()))
    results.append(("svi-quartile-stats", build_svi_quartile_stats_json()))
    results.append(("summary-stats", build_summary_stats_json()))

    print("=" * 60)

    # Summary
    success = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Completed: {success}/{total} data files built")

    if success < total:
        failed = [name for name, r in results if not r]
        print(f"Failed: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
