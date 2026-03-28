# Outcome Study Design: Does Wastewater Monitoring Improve Health Outcomes?

**Status:** Pre-registration draft
**Principal Investigator:** [Your name]
**Date:** January 2026

---

## Research Question

**Primary:** Do counties that gain wastewater surveillance experience better respiratory disease outcomes compared to similar counties without monitoring?

**Secondary:**
1. What is the lead time benefit of wastewater surveillance in newly monitored counties?
2. Does the effect vary by Social Vulnerability Index (SVI)?

---

## Study Design

### Design Type: State-Level Difference-in-Differences with County-Weighted Exposure

**Data constraint:** County-level hospitalization data ended May 2024. We use state-level hospitalization data with a county-weighted treatment intensity measure.

### Treatment Definition
- **Treatment intensity (continuous):** Proportion of state population covered by NWSS monitoring
- **Treatment change:** Change in coverage proportion during study period
- **High-expansion states:** States that increased coverage by >20% or added >5 sites
- **Low-expansion states:** States with stable monitoring (control-like)

### Modified DiD Approach
Instead of binary treated/control at county level, we use:
1. **State-level outcomes:** Hospitalizations per 100k from NHSN
2. **County-weighted exposure:** % of state population in monitored counties
3. **Continuous treatment:** Exploit variation in expansion intensity across states

### Key Assumption
Parallel trends: In the absence of monitoring expansion, high-expansion and low-expansion states would have followed similar hospitalization trajectories.

---

## Data Sources (All Public)

### 1. Wastewater Monitoring Data
- **Source:** CDC NWSS Public Data (Socrata API)
- **Variables:** Site location, county FIPS, first sample date, sample frequency
- **Use:** Calculate state-level coverage intensity over time

### 2. Hospitalization Outcomes (PRIMARY)
- **Source:** CDC NHSN Weekly Hospital Respiratory Data
- **API:** Already integrated in your pipeline (`scripts/fetch_nhsn_data.py`)
- **Variables:** COVID, Flu, RSV hospitalizations by state and week
- **Time period:** 2024-present (with Nov 2024+ being most complete)
- **URL:** https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh

### 3. Treatment Intensity Calculation
For each state-week, calculate:
```
Coverage_st = Σ(county pop with monitoring) / State population
```
This creates a continuous treatment variable ranging from 0 to 1.

### 3. Covariates
- **SVI:** CDC/ATSDR Social Vulnerability Index 2022
- **Urban/Rural:** USDA Rural-Urban Continuum Codes 2023
- **Population:** Census Bureau estimates
- **Vaccination:** CDC county-level vaccination rates
- **Weather:** NOAA climate data (optional)

---

## Identification Strategy

### State-Level DiD Specification

```
Y_st = α_s + γ_t + β × Coverage_st + X_st'δ + ε_st
```

Where:
- `Y_st` = Outcome (hospitalizations per 100k) in state s, week t
- `α_s` = State fixed effects (absorbs time-invariant state characteristics)
- `γ_t` = Week fixed effects (absorbs national trends)
- `Coverage_st` = Proportion of state population with wastewater monitoring
- `X_st` = Time-varying covariates (vaccination rates, etc.)
- `β` = **Effect of 1 percentage point increase in coverage**

### Interpretation
If β = -0.5, then a 10 percentage point increase in coverage is associated with 5 fewer hospitalizations per 100k.

### Event Study (State-Level)
For states with discrete expansion events, estimate:
```
Y_st = α_s + γ_t + Σ_k β_k × 1(t - E_s = k) × HighExpansion_s + ε_st
```

Where `E_s` is the primary expansion event for state s, and `HighExpansion_s` = 1 for states that expanded significantly.

### Robustness: Synthetic Control
For large-expansion states (NY, TX, UT), construct synthetic control states from non-expanders to estimate counterfactual trajectories.

---

## Outcome Variables

### Primary Outcome
- **Total respiratory hospitalizations per 100,000** (weekly, state-level)

### Secondary Outcomes
- COVID-19 hospitalizations per 100,000
- Influenza hospitalizations per 100,000
- RSV hospitalizations per 100,000

### Heterogeneity Tests
- Effect by state SVI (do high-SVI states benefit more from expansion?)
- Effect by baseline coverage (do states starting from zero benefit more?)
- Effect by expansion speed (rapid vs gradual rollout)

### Mechanism Outcomes (if data available)
- Days from wastewater signal to public health response
- Days from wastewater signal to hospitalization peak
- Public health intervention timing (advisories, closures)

---

## Sample and Power

### Sample Size
- **States:** 51 (50 states + DC)
- **High-expansion states:** ~28 (expanded >20% or +5 sites)
- **Low-expansion states:** ~23 (stable monitoring)
- **Time periods:** ~78 weeks (March 2024 - September 2025)

### Power Calculation
With 51 states and 78 weeks:
- Minimum detectable effect: ~8-10% reduction in hospitalizations
- Assuming α=0.05, power=0.80
- **Note:** State-level analysis has less power than county-level

### Matching (Optional)
Match high-expansion to low-expansion states on:
- Pre-treatment hospitalization trends
- State-level SVI
- Population size
- Baseline coverage level

---

## Threats to Validity

### 1. Selection into Treatment
**Threat:** Counties that get monitoring may be systematically different
**Mitigation:**
- County fixed effects absorb time-invariant differences
- Event study tests for pre-trends
- Match on pre-treatment characteristics

### 2. Spillovers
**Threat:** Unmonitored counties may benefit from nearby monitored counties
**Mitigation:**
- Exclude counties adjacent to monitored counties
- Test for spatial spillovers explicitly

### 3. Concurrent Policies
**Threat:** States expanding monitoring may also expand other interventions
**Mitigation:**
- State-by-time fixed effects
- Control for state-level vaccination campaigns, mandates

### 4. Anticipation Effects
**Threat:** Counties may change behavior before monitoring comes online
**Mitigation:**
- Test for pre-trends in event study
- Exclude first 4 weeks post-treatment (burn-in period)

### 5. Measurement Error in Treatment Timing
**Threat:** First sample date may not equal "effective" monitoring date
**Mitigation:**
- Use 12-week lagged treatment (after burn-in)
- Sensitivity analysis with different lag structures

---

## Analysis Plan

### Step 1: Assemble Dataset
1. Pull NWSS data, identify first sample date by county
2. Assemble county-level hospitalization data (or use state-level)
3. Merge with SVI, RUCC, population data
4. Create treatment indicator and event-time variables

### Step 2: Descriptive Analysis
1. Compare treated vs control counties at baseline
2. Plot raw outcome trends for treated vs control
3. Test for parallel pre-trends

### Step 3: Main Analysis
1. Estimate staggered DiD using Callaway-Sant'Anna
2. Report event study plot
3. Report average treatment effect on treated (ATT)

### Step 4: Heterogeneity Analysis
1. Effect by SVI quartile (does monitoring help vulnerable counties more?)
2. Effect by urban/rural status
3. Effect by treatment intensity (sites per capita)

### Step 5: Robustness Checks
1. Alternative control groups (matched vs all)
2. Alternative estimators (TWFE, Sun-Abraham)
3. Placebo tests (fake treatment timing)
4. Sensitivity to treatment lag

---

## Expected Outputs

### If Effect is Found (β < 0, significant)
- "Counties gaining wastewater monitoring experienced X% fewer hospitalizations"
- "The effect is larger in high-SVI counties, suggesting equity benefits"
- Can now calculate ROI: (hospitalizations prevented × cost per hospitalization) / monitoring cost

### If No Effect is Found (β ≈ 0)
- "We find no evidence that monitoring alone improves outcomes"
- "The benefit may require complementary public health response capacity"
- "Infrastructure is necessary but not sufficient"
- Still valuable: tells policymakers what ELSE is needed

### If Effect is Ambiguous
- Report confidence intervals honestly
- Discuss data limitations (county-level data availability)
- Call for better data infrastructure

---

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Data assembly | 4-6 weeks | Identify sources, pull data, merge datasets |
| Descriptive analysis | 2 weeks | Balance tables, trend plots, pre-trend tests |
| Main analysis | 4 weeks | Estimate DiD, event studies, robustness |
| Write-up | 4 weeks | Draft paper, create figures, peer feedback |
| **Total** | **14-16 weeks** | |

---

## Pre-Registration

This study design should be pre-registered at:
- **OSF Registries:** https://osf.io/registries
- **AEA RCT Registry:** https://www.socialscienceregistry.org (if quasi-experimental accepted)
- **PROSPERO:** (for systematic reviews, less relevant here)

Pre-registration protects against:
- Accusations of p-hacking
- Specification searching
- HARKing (hypothesizing after results known)

---

## Data Availability Checklist

Before proceeding, verify access to:

- [ ] NWSS site-level data with first sample dates
- [ ] County-level hospitalization data (at least one source)
- [ ] County FIPS codes for merging
- [ ] SVI 2022 data
- [ ] RUCC 2023 data
- [ ] Computational resources for DiD estimation

---

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*.

---

**Document created:** January 2026
**Status:** Draft for review
**Next step:** Verify data availability, then pre-register
