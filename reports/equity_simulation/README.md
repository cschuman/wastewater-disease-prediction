# Wastewater Surveillance Equity Simulation

This directory contains the complete analysis quantifying what equitable wastewater surveillance coverage would require across the United States.

## Overview

We analyzed 3,144 US counties to identify the equity gap in wastewater disease surveillance and modeled three investment scenarios to close it.

**Key Finding:** High-SVI (Social Vulnerability Index) counties have 44% fewer monitoring sites per capita than low-SVI counties, with 1,259 high-SVI counties having ZERO coverage.

**Recommended Action:** $172M investment to deploy 1,720 new monitoring sites in high-SVI counties over 3-5 years.

## Files in This Directory

### Policy Documents

1. **`EXECUTIVE_SUMMARY.md`**
   - High-level overview of findings
   - Three investment scenarios compared
   - Implementation roadmap
   - Cost-benefit analysis
   - **Audience:** Policymakers, administrators, general readers

2. **`POLICY_BRIEF.md`**
   - Detailed policy recommendations
   - State and county-level priorities
   - Implementation framework
   - Stakeholder engagement plan
   - Immediate action steps
   - **Audience:** Congressional staff, federal agencies, state officials

### Data Files

3. **`county_priority_rankings.csv`** (3,144 rows)
   - **All US counties** with priority scores
   - Columns:
     - `fips`: County FIPS code
     - `state`: State abbreviation
     - `county_name`: County name
     - `population`: Total population
     - `svi_overall`: Social Vulnerability Index (0-1, higher = more vulnerable)
     - `svi_quartile`: SVI quartile classification
     - `n_sites`: Current number of monitoring sites
     - `coverage_pct`: % of population currently covered
     - `priority_score`: Composite priority score (0-100)
     - `priority_tier`: Priority tier (1=highest, 4=lowest)

4. **`top_priority_counties.csv`** (50 rows)
   - **Top 50 counties** for immediate investment
   - Subset of rankings file with highest priorities
   - Use this for Phase 1 implementation planning

5. **`scenario_comparison.csv`** (3 rows)
   - Summary of three investment scenarios
   - Columns: Scenario, New Sites, Setup Cost, 5-Year Cost, Focus
   - **Scenarios:**
     - A: Full equity (equalize all quartiles)
     - B: Priority high-SVI (recommended)
     - C: 50% coverage floor (universal minimum)

## How to Use These Files

### For Policy Analysis

**Question:** "Which counties should get funding first?"
```bash
# View top 20 priority counties
head -21 top_priority_counties.csv
```

**Question:** "How much would it cost to address my state?"
```python
import pandas as pd
df = pd.read_csv('county_priority_rankings.csv')
state_summary = df[df['state'] == 'TX'].agg({
    'priority_score': 'mean',
    'n_sites': 'sum',
    'population': 'sum'
})
print(state_summary)
```

### For Budget Planning

**Question:** "What's the total investment needed?"
- See `scenario_comparison.csv`
- Scenario B (recommended): $172M setup, $602M total over 5 years

**Question:** "How many sites does my county need?"
```python
import pandas as pd
df = pd.read_csv('county_priority_rankings.csv')
# Example: Find Maricopa County, AZ
county = df[(df['state'] == 'AZ') & (df['county_name'] == 'Maricopa County')]
print(county[['county_name', 'priority_score', 'n_sites', 'priority_tier']])
```

### For Geographic Analysis

**Question:** "Which states have the most high-priority counties?"
```python
import pandas as pd
df = pd.read_csv('county_priority_rankings.csv')

# Count Tier 1 counties by state
tier1 = df[df['priority_tier'] == 'Tier 1 (Highest)']
state_counts = tier1.groupby('state').size().sort_values(ascending=False)
print(state_counts.head(10))
```

### For Visualizations

Generate charts and maps:
```bash
# Install dependencies (if needed)
python -m pip install matplotlib seaborn geopandas

# Run visualization script
python ../../src/visualization/equity_simulation_viz.py
```

This will create:
- `visualizations/scenario_comparison.png` - Scenario cost comparison
- `visualizations/priority_counties_viz.png` - Top priority counties
- `visualizations/state_investment_priorities.png` - State-level priorities
- `visualizations/coverage_gap_analysis.png` - Current equity gap
- `visualizations/executive_summary.png` - One-page summary viz

## Analysis Methodology

### Data Sources
- **NWSS (National Wastewater Surveillance System):** 500,000 samples from 684 counties (Jan 11, 2026)
- **CDC/ATSDR SVI 2022:** Social Vulnerability Index for 3,144 US counties
- **US Census:** County population estimates

### Priority Scoring Algorithm

Each county receives a composite priority score (0-100) based on:

1. **SVI Score (35% weight)**
   - Higher vulnerability = higher priority
   - Normalized to 0-100 scale

2. **Coverage Gap (30% weight)**
   - Expected sites (based on national average) minus current sites
   - Normalized to 0-100 scale

3. **Population Impact (15% weight)**
   - Larger population = more people benefit
   - Log-scaled to prevent overwhelming effect of mega-counties

4. **Zero Coverage Bonus (20% weight)**
   - Counties with zero monitoring get +20 points
   - Ensures no community is completely unserved

**Formula:**
```
priority_score = (svi_norm × 0.35) + (gap_norm × 0.30) + (pop_norm × 0.15) + (zero_coverage × 20)
```

### Scenario Modeling

**Scenario A: Full Equity**
- Target: Match low-SVI (Q1) sites-per-capita rate (4.79/million)
- Method: Calculate needed sites for each county to reach target
- Result: 799 new sites, $280M over 5 years

**Scenario B: Targeted Priority (Recommended)**
- Target: Minimum coverage in all high-SVI (Q3/Q4) counties
- Method: Population-based minimum (1 site per 100k, 2 per 500k, etc.)
- Result: 1,720 new sites, $602M over 5 years

**Scenario C: Universal Minimum**
- Target: 50% population coverage in every county
- Method: Calculate sites needed to reach 50% threshold (assumes 100k coverage per site)
- Result: 2,845 new sites, $996M over 5 years

### Cost Assumptions

Based on CDC/EPA guidance and academic literature:
- **Setup cost per site:** $100,000 (equipment, lab, protocols)
- **Annual operating cost:** $50,000 (testing, staffing, QA/QC)
- **Projection horizon:** 5 years

## Reproducing the Analysis

### Requirements
```bash
# Python 3.10+
pip install pandas numpy scipy statsmodels pyarrow
```

### Run Full Analysis
```bash
cd /path/to/wastewater-disease-prediction
python src/analysis/equity_simulation.py
```

### Expected Output
```
================================================================================
WASTEWATER SURVEILLANCE EQUITY SIMULATION
Modeling What Equitable Coverage Would Require
================================================================================

[1/6] Loading data...
[2/6] Analyzing current coverage by SVI...
[3/6] Running equity scenarios...
[4/6] Creating priority investment map...
[5/6] Generating policy recommendations...
[6/6] Saving outputs...

Outputs saved to reports/equity_simulation/
================================================================================
```

## Key Findings Summary

### Current State
- **3,144 counties** analyzed
- **678 (21.6%)** have wastewater monitoring
- **1,259 high-SVI counties** have **ZERO** coverage
- **44% disparity** in sites per capita (high vs low SVI)

### Recommended Investment (Scenario B)
- **1,720 new sites** needed
- **$172M setup** + $430M operating over 5 years
- **3-5 year timeline**
- **Prioritizes 1,259 zero-coverage high-SVI counties**

### Top Priority States (by new sites needed)
1. Texas - 225 sites
2. California - 157 sites
3. Georgia - 126 sites
4. Florida - 97 sites
5. Mississippi - 76 sites

### Top Priority Counties
1. Maricopa County, AZ (4.4M pop, SVI 0.71, 0 sites)
2. Bexar County, TX (2.0M pop, SVI 0.92, 0 sites)
3. San Bernardino County, CA (2.2M pop, SVI 0.89, 0 sites)
4. Riverside County, CA (2.4M pop, SVI 0.82, 0 sites)
5. Broward County, FL (1.9M pop, SVI 0.78, 0 sites)

## Questions & Contact

**For questions about the analysis:**
- Review the code: `../../src/analysis/equity_simulation.py`
- Check the methodology section above
- See the technical documentation in the code comments

**For policy questions:**
- See `POLICY_BRIEF.md` for implementation framework
- See `EXECUTIVE_SUMMARY.md` for high-level overview

**For data access:**
- All CSV files are ready to use
- Python/R/Excel compatible
- No special software required

## Citation

If using this analysis in publications or policy documents:

```
Wastewater Surveillance Equity Simulation (2026). County-level analysis of
monitoring coverage disparities by social vulnerability. Data Science Team,
Public Health Wastewater Surveillance Project.
```

## Updates & Versions

- **Version 1.0** (January 11, 2026): Initial analysis with NWSS data through Jan 11, 2026
- **Data currency:** NWSS updated regularly; re-run analysis quarterly for latest

---

**Last Updated:** January 11, 2026
**Analysis Code:** `../../src/analysis/equity_simulation.py`
**Visualization Code:** `../../src/visualization/equity_simulation_viz.py`
