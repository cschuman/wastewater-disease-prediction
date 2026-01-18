# Expert Panel Review: Wastewater Disease Prediction Project

**Review Date:** January 2026
**Review Type:** Pre-submission critical analysis
**Purpose:** Identify unsupported claims before peer review

---

## Panel Composition

| Expert | Role | Affiliation Type |
|--------|------|------------------|
| Dr. A | **Epidemiologist** | CDC/State Health Department |
| Dr. B | **Health Economist** | Health Policy Institute |
| Dr. C | **Biostatistician** | Academic Medical Center |
| Dr. D | **Environmental Health Scientist** | School of Public Health |
| Dr. E | **Health Equity Researcher** | Social Determinants Center |

---

## Executive Summary

**Overall Assessment:** The project makes a **well-supported infrastructure equity claim** but has conflated this with **unsupported health outcome and ROI claims**. The core finding (surveillance infrastructure disparity) is rigorous. The leap to health impact is not.

| Claim Category | Verdict | Action Required |
|----------------|---------|-----------------|
| Infrastructure gap exists | **SUPPORTED** | None |
| Gap is statistically significant | **SUPPORTED** | None |
| Cost estimates are reasonable | **SUPPORTED** | Minor caveat needed |
| Adding sites improves outcomes | **NOT SUPPORTED** | Remove or qualify |
| ROI of $5-15 per $1 | **NOT SUPPORTED** | ~~Removed~~ |
| 141M people "get early detection" | **MISLEADING** | Reframe |
| 10-100x more cost-effective | **NOT SUPPORTED** | Remove or cite |

---

## Claim-by-Claim Analysis

### CLAIM 1: "High-SVI counties have 33% fewer monitoring sites per capita"

**Source:** `health_equity_findings.md`, multiple reports

#### Epidemiologist (Dr. A):
> **VERDICT: SUPPORTED**
>
> The methodology is sound. Using CDC SVI quartiles and comparing sites per capita is standard practice. The ANOVA finding (p=0.0001) is robust.
>
> Minor concern: "33% fewer" is calculated from Q1 vs Q4 comparison. The linear relationship across all quartiles would be a cleaner metric.

#### Biostatistician (Dr. C):
> **VERDICT: SUPPORTED**
>
> Statistical approach is appropriate:
> - ANOVA for quartile comparison: p=0.0001 ✓
> - OLS regression with controls: p=0.007 after urban/rural adjustment ✓
> - Within-state analysis showing state-level effect: appropriate ✓
>
> The urban/rural confounding analysis strengthens the claim. The finding that the effect is concentrated in urban counties (p=0.0002) actually makes the equity argument STRONGER.

#### Health Equity Researcher (Dr. E):
> **VERDICT: SUPPORTED**
>
> This is exactly the kind of infrastructure disparity analysis the field needs. The stratified finding (effect in urban but not rural counties) suggests this is about resource allocation decisions, not just rural infrastructure limitations.
>
> Suggestion: Frame this as "differential investment" rather than just "gap" - it implies policy choices, not natural variation.

**PANEL CONSENSUS: CLAIM SUPPORTED - No changes needed**

---

### CLAIM 2: "1,259 high-SVI counties have zero surveillance coverage"

**Source:** `equity_simulation.py`, multiple reports

#### Epidemiologist (Dr. A):
> **VERDICT: SUPPORTED**
>
> This is a count from the data. Verifiable.
>
> Caveat: "Zero coverage" means no NWSS sites. Some may have state/local programs not in NWSS. Recommend adding: "as reported to NWSS"

#### Biostatistician (Dr. C):
> **VERDICT: SUPPORTED**
>
> This is descriptive statistics, not inferential. The count is what it is.

**PANEL CONSENSUS: CLAIM SUPPORTED - Add caveat about NWSS-reported data**

---

### CLAIM 3: "$172M investment could close the infrastructure gap"

**Source:** `equity_simulation.py`, `EXECUTIVE_SUMMARY.md`

#### Health Economist (Dr. B):
> **VERDICT: SUPPORTED WITH CAVEATS**
>
> The cost calculation is transparent:
> - $100k per site setup (CDC/EPA guidance range: $50k-$150k) ✓
> - $50k annual operating (reasonable estimate) ✓
> - 1,720 sites × costs = $172M setup, $602M 5-year total ✓
>
> **Caveats needed:**
> 1. Costs assume existing wastewater treatment plant infrastructure
> 2. Rural areas may have higher per-site costs (travel, logistics)
> 3. Operating costs may be higher in areas with less existing lab capacity
> 4. No inflation adjustment shown
>
> Recommendation: Present as "estimated $172M" with uncertainty range ($140M-$220M)

#### Environmental Health Scientist (Dr. D):
> **VERDICT: SUPPORTED WITH CAVEATS**
>
> The per-site cost estimates are within reasonable range based on NWSS implementation experience.
>
> **Technical concerns:**
> 1. Not all high-SVI counties have suitable wastewater treatment plants
> 2. Some areas rely heavily on septic systems (especially rural high-SVI)
> 3. Small WWTPs may not produce representative samples
>
> Recommendation: Add feasibility assessment layer to county prioritization

**PANEL CONSENSUS: CLAIM SUPPORTED - Add uncertainty ranges and feasibility caveats**

---

### CLAIM 4: ~~"For every $1 invested, save $5-15 in outbreak response costs"~~ (REMOVED)

**Source:** Originally in `EXECUTIVE_SUMMARY.md`, `POLICY_BRIEF.md`, `+page.svelte`

#### Health Economist (Dr. B):
> **VERDICT: NOT SUPPORTED**
>
> This is the most problematic claim. To calculate ROI, you need:
>
> 1. **Counterfactual:** What happens WITHOUT the investment?
>    - Not estimated in this analysis
>
> 2. **Causal pathway:** Investment → Detection → Intervention → Outcomes
>    - No link between site density and detection quality established
>    - No link between detection and intervention established
>    - No link between intervention and outcomes established
>
> 3. **Outcome valuation:** What is a prevented hospitalization worth?
>    - Not calculated
>
> 4. **Attribution:** What fraction of improvement is due to wastewater vs other factors?
>    - Not addressed
>
> **This claim appears to be aspirational, possibly derived from general WBE literature, not from this project's analysis.**
>
> ~~RECOMMENDATION: REMOVE ENTIRELY or replace with "ROI requires further study"~~
>
> **UPDATE:** Claim has been removed from all documents. Good.

#### Biostatistician (Dr. C):
> **VERDICT: NOT SUPPORTED**
>
> The fundamental problem: This project establishes CORRELATION (SVI ↔ coverage) but claims CAUSATION (investment → outcomes).
>
> To support ROI claims, you would need:
> - Randomized deployment study, OR
> - Difference-in-differences with outcome data, OR
> - Regression discontinuity design
>
> None of these are present. The DiD analysis in the project looks at signal QUALITY, not health OUTCOMES.

**PANEL CONSENSUS: CLAIM NOT SUPPORTED - Correctly removed**

---

### CLAIM 5: "Early detection for 141M people in underserved areas"

**Source:** `EXECUTIVE_SUMMARY.md`, `+page.svelte`

#### Epidemiologist (Dr. A):
> **VERDICT: MISLEADING - REFRAME**
>
> This conflates "population in underserved areas" with "people who will receive early detection."
>
> What we can say: "141M people live in high-SVI counties currently without NWSS monitoring"
>
> What we CANNOT say: "These people will get early detection if we add sites"
>
> The leap requires assuming:
> 1. New sites will produce usable signals (burn-in period noted in analysis)
> 2. Signals will be acted upon by public health authorities
> 3. Actions will reach the 141M people in time to matter
>
> None of these are established.

#### Health Equity Researcher (Dr. E):
> **VERDICT: MISLEADING - REFRAME**
>
> "Early detection for 141M people" implies a benefit that hasn't been demonstrated.
>
> Better framing: "141M people live in counties that would gain surveillance infrastructure"
>
> The distinction matters because:
> - Infrastructure ≠ utilization
> - Surveillance data ≠ public health response
> - Response ≠ community benefit
>
> This is a common problem in infrastructure equity work - we conflate "access" with "benefit."

**PANEL CONSENSUS: REFRAME as infrastructure access, not health benefit**

**Current language (after edits):** "141M people live in underserved high-SVI areas without monitoring" - ACCEPTABLE

---

### CLAIM 6: ~~"Wastewater surveillance is 10-100x more cost-effective than clinical testing"~~

**Source:** Originally in `EXECUTIVE_SUMMARY.md`

#### Health Economist (Dr. B):
> **VERDICT: NOT SUPPORTED IN THIS ANALYSIS**
>
> This is likely a literature citation (probably from COVID-era comparisons) but:
>
> 1. No citation provided
> 2. Not calculated from this project's data
> 3. "Cost-effective" requires outcome measurement (not just cost comparison)
> 4. The 10-100x range is suspiciously wide, suggesting uncertainty or context-dependence
>
> If citing literature, provide the citation. If claiming for this context, provide the calculation.

#### Environmental Health Scientist (Dr. D):
> **VERDICT: CONTEXT-DEPENDENT**
>
> The 10-100x figure likely comes from per-test cost comparisons during COVID:
> - Clinical PCR: $50-150 per test
> - Wastewater: $100-500 per sample covering thousands of people
>
> But "cost-effective" ≠ "cheaper per test." Cost-effectiveness requires:
> - Comparable sensitivity/specificity
> - Comparable actionability
> - Outcome measurement
>
> Wastewater provides DIFFERENT information than clinical testing, not necessarily BETTER information per dollar.

**PANEL CONSENSUS: REMOVE unless cited, or replace with "literature suggests potential cost advantages"**

**Status:** Should be removed or cited in remaining documents.

---

### CLAIM 7: "High-SVI states receive ~1.8 fewer days of early warning"

**Source:** `health_equity_findings.md`

#### Epidemiologist (Dr. A):
> **VERDICT: CORRECTLY QUALIFIED**
>
> The analysis appropriately notes:
> - p=0.78 (NOT statistically significant)
> - Selection bias in peak detection
> - Limited statistical power
>
> This is an example of GOOD scientific communication - reporting a finding while acknowledging its limitations.

#### Biostatistician (Dr. C):
> **VERDICT: CORRECTLY QUALIFIED**
>
> The analysis does the right thing by:
> 1. Reporting the point estimate (1.8 days)
> 2. Reporting the p-value (0.78)
> 3. Explaining why the estimate may be biased
> 4. Not overclaiming based on a non-significant result
>
> This should be a model for other claims in the document.

**PANEL CONSENSUS: CLAIM APPROPRIATELY QUALIFIED - No changes needed**

---

### CLAIM 8: "The gap persists after controlling for urban/rural differences"

**Source:** `health_equity_findings.md`

#### Biostatistician (Dr. C):
> **VERDICT: SUPPORTED**
>
> The confounding analysis is well-executed:
> - Baseline: β=-0.811, p=0.022
> - With RUCC control: β=-0.721, p=0.007
> - Coefficient change: 11% (modest confounding)
> - Stratified analysis shows effect in urban (p=0.0002) but not rural (p=0.52)
>
> This actually STRENGTHENS the equity argument by showing it's not just rural infrastructure limitations.

#### Health Equity Researcher (Dr. E):
> **VERDICT: SUPPORTED**
>
> The stratified finding is important: within urban areas, high-SVI counties still have fewer sites. This suggests policy/resource allocation bias, not just geographic constraints.

**PANEL CONSENSUS: CLAIM SUPPORTED**

---

### CLAIM 9: "Reduced outbreak response costs ($50-500M per major outbreak)"

**Source:** `POLICY_BRIEF.md`, `EXECUTIVE_SUMMARY.md`

#### Health Economist (Dr. B):
> **VERDICT: NOT SUPPORTED BY THIS ANALYSIS**
>
> The $50-500M figure is likely from general outbreak cost literature, but:
>
> 1. Not calculated for this specific context
> 2. Not attributed to wastewater surveillance specifically
> 3. Assumes detection → response → cost savings pathway that isn't validated
>
> The range is so wide (10x) that it's essentially meaningless for decision-making.

#### Epidemiologist (Dr. A):
> **VERDICT: PLAUSIBLE BUT UNVALIDATED**
>
> Major outbreaks do cost tens to hundreds of millions. But attributing savings to wastewater surveillance requires demonstrating:
>
> 1. The outbreak would have been detected LATER without wastewater
> 2. Earlier detection led to DIFFERENT response
> 3. Different response led to REDUCED costs
>
> None established here.

**PANEL CONSENSUS: REMOVE or explicitly label as "literature estimate, not validated in this analysis"**

---

## Summary: Claims Requiring Action

### Remove or Already Removed:
- [x] "$5-15 saved per $1 invested" - REMOVED
- [ ] "10-100x more cost-effective than clinical testing" - REMOVE unless cited
- [ ] "$50-500M per major outbreak" - REMOVE or cite

### Reframe:
- [x] "Early detection for 141M people" → "141M people in areas without monitoring" - DONE

### Add Caveats:
- [ ] Cost estimates need uncertainty ranges
- [ ] Infrastructure feasibility limitations
- [ ] "as reported to NWSS" qualifier

### No Changes Needed:
- 33% fewer sites per capita (supported)
- 1,259 counties with zero coverage (supported)
- Statistical significance (supported)
- Urban/rural confounding analysis (supported)
- Early warning penalty correctly qualified (supported)

---

## Recommendations for Peer Review Submission

### 1. Reframe the Contribution
**Current framing:** "This analysis shows wastewater surveillance investment will save money and lives"

**Honest framing:** "This analysis documents a statistically significant infrastructure equity gap and estimates the cost to close it. Whether closing the gap improves health outcomes requires further study."

### 2. Separate What You Proved from What You Hypothesize
Create explicit sections:
- **Validated findings** (infrastructure gap, costs)
- **Hypothesized benefits** (better detection, reduced burden)
- **Research needed** (longitudinal outcome studies)

### 3. Add Limitations Section
Current limitations section is good but should add:
- "This analysis does not establish causal links between surveillance coverage and health outcomes"
- "ROI and cost-effectiveness require outcome data not available in this study"

### 4. Cite Literature Claims
Any claim about wastewater surveillance effectiveness from general literature should be cited, not presented as findings from this analysis.

### 5. Pre-Register Future Analysis
If you plan to study outcomes, pre-register the analysis plan. This protects against accusations of p-hacking when the outcome data becomes available.

---

## What This Project DOES Well

The panel wants to acknowledge genuine strengths:

1. **Rigorous infrastructure analysis** - The coverage disparity finding is statistically sound
2. **Good confounding analysis** - Urban/rural adjustment strengthens conclusions
3. **Transparent methodology** - Code is available, methods are documented
4. **Appropriate qualification** - The early warning penalty analysis correctly notes non-significance
5. **Actionable policy focus** - County-level prioritization is useful
6. **Honest cost estimation** - Infrastructure costs are reasonable and documented

**The core contribution is valuable. The problem was overclaiming on outcomes.**

---

## Checklist for Peer Review Readiness

- [x] Remove unsupported ROI claims
- [x] Reframe "early detection for 141M" as infrastructure access
- [x] Add uncertainty ranges to cost estimates (added to POLICY_BRIEF.md)
- [x] Cite or remove "10-100x cost-effective" claim (removed/qualified)
- [x] Cite or remove "$50-500M per outbreak" claim (removed from claims, noted as literature estimate)
- [x] Add explicit "this analysis does not establish causation" statement (added to health_equity_findings.md)
- [x] Separate validated findings from hypothesized benefits (done in POLICY_BRIEF.md and EXECUTIVE_SUMMARY.md)
- [ ] Add pre-registration plan for future outcome analysis (recommended for future work)

---

**Panel Review Completed:** January 2026
**Recommendation:** Address remaining items, then submit with confidence in the infrastructure equity finding.
