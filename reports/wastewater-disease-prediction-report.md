# Wastewater Surveillance Disease Prediction Project

## Comprehensive Research & Scoping Report

**Project Goal:** Build a predictive model using CDC NWSS wastewater surveillance data to forecast disease outbreaks before they manifest in hospital admission data.

**Report Date:** January 2026

---

## Executive Summary

Wastewater-based epidemiology (WBE) has emerged as a powerful early warning system for disease outbreaks. By detecting pathogen RNA in sewage before infected individuals seek medical care, wastewater surveillance can provide 1-2 weeks of lead time over traditional clinical surveillance. This project proposes building novel predictive models that address key gaps in current research: multi-pathogen combined burden forecasting, rural health equity applications, and spatiotemporal outbreak propagation prediction.

**Key Finding:** While single-pathogen COVID-19 prediction has been well-studied (achieving 10-day lead times with r=0.77-0.89 correlation), **multi-pathogen combined respiratory burden prediction remains unexplored**—despite this being what hospitals actually need for capacity planning.

---

## 1. Data Landscape

### 1.1 Primary Data Sources

#### CDC National Wastewater Surveillance System (NWSS)

| Attribute | Details |
|-----------|---------|
| **Coverage** | ~1,500 monitoring sites across all 50 states, 7 territories, tribal communities |
| **Population Reach** | ~40% of U.S. population |
| **Update Frequency** | Weekly (every Friday) |
| **Historical Depth** | COVID-19: 2020-present; Influenza A: 2022-present; RSV: 2023-present |
| **Access Method** | Public API (Socrata) + CSV download |

**Primary Datasets:**

1. **NWSS Public SARS-CoV-2 Wastewater Metric Data**
   - URL: https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6
   - API Endpoint: `https://data.cdc.gov/resource/2ew6-ywp6.json`
   - Fields: sewershed, date, percentile, percent change, detection flag, population served

2. **NWSS Public SARS-CoV-2 Concentration Data**
   - URL: https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Concentration-in-Wastewater/g653-rqe2
   - Fields: raw concentration values, normalization factors, flow data

3. **CDC GitHub Repository**
   - URL: https://github.com/CDCgov/NWSS
   - Contains: code sharing resources, data dictionaries, analysis tools

#### WastewaterSCAN (Stanford-Emory-Verily Partnership)

| Attribute | Details |
|-----------|---------|
| **Coverage** | 151 sites across 41 states (reduced from 194 in July 2024) |
| **Population Reach** | ~12% of U.S. population with consistent monitoring |
| **Update Frequency** | Variable (research-oriented) |
| **Access** | Stanford Digital Repository (contact required) |

**Expanded Pathogen Panel (as of October 2025):**

| Category | Pathogens Tracked |
|----------|-------------------|
| **Respiratory** | SARS-CoV-2, Influenza A (H1, H3, H5), Influenza B, RSV, HMPV, Enterovirus D68 |
| **Gastrointestinal** | Norovirus GII, Rotavirus, Human adenovirus group F |
| **Emerging** | Candida auris, Hepatitis A, Measles, Mpox (clades Ib and II), West Nile virus |

**Data Access:**
- Stanford Digital Repository: https://purl.stanford.edu/hj801ns5929
- Contact: wwscan_stanford_emory@lists.stanford.edu

#### State-Level Sources

**California Cal-SuWers Network:**
- URL: https://data.chhs.ca.gov/dataset/wastewater-surveillance-data-california
- Tracks: COVID-19, Influenza, RSV, Mpox, Norovirus
- Integrated with CDC NWSS

### 1.2 Hospital/Clinical Target Data

#### NHSN Hospital Respiratory Data (Primary Target Variable)

| Attribute | Details |
|-----------|---------|
| **Mandate** | CMS requires reporting as of November 1, 2024 |
| **Diseases** | COVID-19, Influenza, RSV (lab-confirmed) |
| **Stratification** | Age groups, geographic regions |
| **Granularity** | State/territory level, weekly |
| **Access** | Public API via HealthData.gov |

**Primary Dataset:**
- URL: https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/n3kj-exp9
- Fields: hospital admissions, ICU occupancy, bed capacity, by age group

#### Historical HHS Protect Data (for training)

- COVID-19 Hospital Capacity by State Timeseries: https://catalog.data.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-by-state-timeseries-cf58c
- Note: Data frozen as of May 3, 2024; useful for historical model training

### 1.3 Auxiliary Data Sources

| Data Type | Source | Use Case |
|-----------|--------|----------|
| **Weather/Climate** | NOAA, NWS | Temperature affects viral shedding/decay; precipitation causes dilution |
| **Demographics** | U.S. Census | Population density, age structure, vulnerability indices |
| **Vaccination Rates** | CDC Immunization Data | Modulates transmission and severity |
| **Social Vulnerability** | CDC SVI | Identifies high-risk communities |
| **Variant Proportions** | CDC NWSS Variant Dashboard | Freyja demixing for SARS-CoV-2 variants |

---

## 2. Current State of the Art

### 2.1 Published Lead Time Achievements

| Disease | Lead Time | Correlation | Source |
|---------|-----------|-------------|--------|
| COVID-19 Hospitalizations | 10 days | r = 0.77-0.89 | ScienceDirect 2023 |
| COVID-19 Cases | 0-11 days (avg 6) | Variable | Rural Idaho Study |
| Influenza Outbreaks | 17 days | Significant | Nature Scientific Reports |
| RSV Hospitalizations | 12+ days | Significant | Pediatric RSV study |
| COVID-19 Wave Peaks | Up to 63 days | Early pandemic | Early pandemic studies |

### 2.2 Model Performance Benchmarks

**Key Finding:** Wastewater models improve hospitalization predictions by **11-15%** over case-data-only models.

| Forecast Horizon | Best Models | Performance |
|------------------|-------------|-------------|
| 1-2 weeks | ARIMA, GAM | Most accurate; flexible statistical approaches |
| 3-4 weeks | n-sub-epidemic ensemble | Captures longer-term trends |
| Multi-step | LSTM, Hybrid ARIMA-LSTM | 2.4% MAPE achieved; handles nonlinear dynamics |

**Accuracy Metrics:**
- Predicted vs. observed hospitalizations correlation: r = 0.77-0.89
- Mean absolute error: 4-6 patients/100k population for weekly predictions
- 1-4 week preparation window demonstrated across 159 U.S. counties

### 2.3 Existing Approaches

| Approach | Description | Limitations |
|----------|-------------|-------------|
| **ARIMA with covariates** | Traditional time series with wastewater as exogenous variable | Assumes linearity |
| **GAM (Generalized Additive Models)** | Spline-based seasonality modeling | Single-site focus |
| **XGBoost/Random Forest** | Feature-rich ensemble methods | No temporal structure |
| **LSTM/GRU** | Deep learning for sequences | Requires large datasets |
| **n-sub-epidemic ensemble** | CDC's ensemble forecasting approach | Complex, computationally intensive |

---

## 3. Research Gaps & Novel Opportunities

### 3.1 Gap Analysis

| Gap | Current State | Opportunity |
|-----|---------------|-------------|
| **Multi-pathogen prediction** | Models focus on single pathogens | Hospitals need TOTAL respiratory burden |
| **Rural coverage** | 70% of surveillance is urban/coastal | Transfer learning to underserved areas |
| **Spatiotemporal modeling** | Only 20% use explicit spatial methods | Predict WHERE outbreaks will spread |
| **Variant-severity coupling** | Genomic data underutilized | Predict severity changes from variant emergence |
| **Antimicrobial resistance** | Correlations established only | Predictive AMR models from wastewater |

### 3.2 Technical Challenges

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Sewershed population estimation** | 73% variance between methods | Census block interpolation, flow normalization |
| **Reporting delays** | 1-week data lag typical | Nowcasting layers, temporal interpolation |
| **Stormwater dilution** | Concentration drops during rain | PMMoV normalization, flow adjustment |
| **RNA degradation** | Temperature-dependent decay | Temperature correction factors |
| **Site heterogeneity** | Different WWTPs have different characteristics | Hierarchical/mixed-effects models |

---

## 4. Proposed Novel Projects

### 4.1 Project A: Combined Respiratory Pathogen Hospital Burden Forecasting (RECOMMENDED)

#### Overview

**Prediction Target:** Weekly total respiratory hospitalizations (COVID-19 + Influenza + RSV combined) at the state level, stratified by age group:
- Pediatric (<18 years)
- Adult (18-64 years)
- Elderly (65+ years)

**Why This Matters:** Hospitals don't manage COVID, flu, and RSV separately—they manage bed capacity. A combined model directly addresses operational needs.

#### Input Features

| Category | Features |
|----------|----------|
| **Wastewater signals** | SARS-CoV-2, Influenza A, Influenza B, RSV concentrations (normalized) |
| **Temporal** | Week of year, trend indicators, lagged values (1-4 weeks) |
| **Historical** | Past hospitalization rates by age group |
| **Population** | Vaccination coverage by pathogen, demographic composition |
| **Environmental** | Temperature, precipitation, humidity |
| **Vulnerability** | CDC Social Vulnerability Index |

#### Architecture Options

**Option 1: Multi-Output Regression**
```
Input: [wastewater_covid, wastewater_flu, wastewater_rsv, covariates]
       ↓
   Shared Feature Extraction (XGBoost/Neural Net)
       ↓
   Multi-head Output: [hosp_pediatric, hosp_adult, hosp_elderly]
```

**Option 2: Hierarchical Time Series**
```
Level 1: National total respiratory hospitalizations
Level 2: State-level totals
Level 3: State × Age group
Reconciliation: Bottom-up or optimal combination
```

**Option 3: Multi-Task Learning**
```
Shared encoder for all pathogens/age groups
Task-specific decoders for each output
Auxiliary tasks: individual pathogen predictions (for regularization)
```

#### Evaluation Methodology

**Retrospective Study Period:** 2023-24 and 2024-25 respiratory seasons

**Metrics:**
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- WIS (Weighted Interval Score) for probabilistic forecasts
- 95% Prediction Interval coverage

**Baselines for Comparison:**
1. Sum of independent single-pathogen models
2. Historical seasonal average
3. CDC Respiratory Virus ensemble forecasts

**Validation Strategy:**
- Leave-one-state-out cross-validation
- Rolling-window temporal validation (train on weeks 1-N, predict week N+1)

#### Novelty Statement

> "To our knowledge, no published work has developed a unified predictive model that forecasts combined respiratory hospitalization burden from multi-pathogen wastewater signals. Existing models predict pathogens individually, requiring hospitals to manually aggregate forecasts. Our approach directly models the quantity hospitals need to plan for: total respiratory admissions by age group."

#### Feasibility Assessment

| Dimension | Assessment | Notes |
|-----------|------------|-------|
| **Data availability** | HIGH | All data publicly available from CDC |
| **Computational needs** | MODERATE | Standard ML infrastructure sufficient |
| **Timeline** | 4-6 months | For full retrospective study |
| **Risk** | LOW | Building on established single-pathogen approaches |
| **Publication potential** | HIGH | Novel, timely, actionable |

---

### 4.2 Project B: Rural Health Equity Early Warning System

#### Overview

**Prediction Target:** Binary outbreak alert (yes/no) and continuous risk score for rural counties lacking clinical surveillance infrastructure, with 2-week lead time before hospitalization surge.

**Why This Matters:** 70% of wastewater surveillance infrastructure is in urban/coastal areas, yet rural communities face 2x higher health access barriers.

#### Input Features

| Category | Features |
|----------|----------|
| **Transferred wastewater signals** | Distance-weighted signals from nearest monitored sewersheds |
| **Geographic** | Distance to nearest monitoring site, transportation connectivity |
| **Demographic** | Rural/urban classification, population density, age distribution |
| **Healthcare access** | Hospital beds per capita, distance to nearest hospital |
| **Vulnerability** | CDC SVI, insurance coverage rates |
| **Historical** | Past outbreak timing patterns in region |

#### Methodology

1. **Signal Transfer Model:** Learn how wastewater signals propagate from monitored urban areas to unmonitored rural areas
2. **Feature Engineering:** Create distance-decay weighted features from surrounding monitored sites
3. **Classification:** Binary alert (surge imminent yes/no) with calibrated probability
4. **Regression:** Days until surge (for positive alerts)

#### Evaluation

- Retrospective analysis on known rural outbreak events (2022-2025)
- Sensitivity, Specificity, PPV at 2-week horizon
- Equity analysis: Does model reduce urban-rural disparity in detection?

#### Novelty Statement

> "This work represents the first systematic approach to transfer wastewater surveillance signals from monitored urban sewersheds to predict outbreak timing in unmonitored rural communities, directly addressing documented health equity gaps in wastewater-based epidemiology."

#### Feasibility Assessment

| Dimension | Assessment | Notes |
|-----------|------------|-------|
| **Data availability** | MODERATE | Requires careful linking of NWSS to rural county outcomes |
| **Computational needs** | LOW-MODERATE | Spatial interpolation, standard classification |
| **Timeline** | 6-8 months | Including data curation |
| **Risk** | MODERATE | Signal transfer efficacy unknown |

---

### 4.3 Project C: Spatiotemporal Outbreak Propagation with Graph Neural Networks

#### Overview

**Prediction Target:** For each sewershed node in a state-level graph, predict:
1. Probability of significant viral concentration increase (>50% week-over-week) in next 2 weeks
2. Expected timing of peak viral concentration (within 4-week window)
3. Direction and strength of predicted spread to connected sewersheds

#### Architecture

```
Graph Construction:
- Nodes: Sewersheds (with features: concentrations, population, demographics)
- Edges: Geographic proximity, estimated mobility flow, route connectivity

Model:
- Graph Attention Network (GATv2) for spatial message passing
- Temporal fusion via GRU or Transformer encoder
- Multi-task output heads for probability, timing, and direction
```

#### Pilot Scope

Focus on single state with dense coverage:
- California: 68 NWSS sites
- Texas: 51 NWSS sites

#### Novelty Statement

> "While graph neural networks have revolutionized spatial prediction in other domains, only 20% of wastewater epidemiology studies use explicit spatiotemporal modeling. This work represents the first application of GNNs to wastewater outbreak propagation prediction, addressing the critical question of 'where will disease spread' rather than just 'when will cases rise.'"

#### Feasibility Assessment

| Dimension | Assessment | Notes |
|-----------|------------|-------|
| **Data availability** | HIGH | NWSS provides site-level data with coordinates |
| **Computational needs** | MODERATE-HIGH | GPU training for GNN |
| **Timeline** | 8-12 months | Full development and validation |
| **Risk** | MODERATE | Graph structure definition requires careful design |

---

## 5. Technical Implementation Guide

### 5.1 Data Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CDC NWSS API   │────▶│  Data Ingestion │────▶│  Feature Store  │
│  (weekly pull)  │     │  & Validation   │     │  (PostgreSQL)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐            │
│  NHSN Hospital  │────▶│  Target Variable│────────────┤
│  Data API       │     │  Processing     │            │
└─────────────────┘     └─────────────────┘            │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Weather/Demo   │────▶│  Covariate      │────▶│  Training       │
│  APIs           │     │  Engineering    │     │  Pipeline       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 5.2 Recommended Tech Stack

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Data storage** | PostgreSQL + TimescaleDB | Time series optimization |
| **Data pipeline** | Apache Airflow or Prefect | Scheduled weekly pulls |
| **ML framework** | PyTorch + PyTorch Forecasting | Flexible, GNN support |
| **Time series** | statsmodels, Prophet, Darts | Established baselines |
| **Experiment tracking** | MLflow or Weights & Biases | Reproducibility |
| **Deployment** | FastAPI + Docker | API serving |

### 5.3 Key Python Libraries

```python
# Data
pandas, numpy, geopandas
requests, sodapy  # Socrata API client

# Time Series
statsmodels  # ARIMA, SARIMAX
prophet      # Facebook's forecasting
darts        # Unified forecasting API
sktime       # Scikit-learn compatible

# Machine Learning
scikit-learn
xgboost, lightgbm
pytorch, pytorch-forecasting
pytorch-geometric  # For GNN

# Evaluation
properscoring  # WIS, CRPS
```

---

## 6. Timeline & Milestones

### Project A: Combined Burden Forecasting (Recommended)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Data Acquisition** | Weeks 1-3 | Automated data pipeline, cleaned datasets |
| **Phase 2: EDA & Feature Engineering** | Weeks 4-6 | Feature importance analysis, correlation studies |
| **Phase 3: Baseline Models** | Weeks 7-10 | ARIMA, XGBoost baselines with metrics |
| **Phase 4: Multi-Output Models** | Weeks 11-16 | Neural net architectures, hyperparameter tuning |
| **Phase 5: Validation & Analysis** | Weeks 17-20 | Cross-validation, ablation studies |
| **Phase 6: Documentation** | Weeks 21-24 | Technical report, potential publication draft |

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Data quality issues** | Medium | High | Implement robust validation; use multiple data sources |
| **Weak multi-pathogen signal** | Low | High | Fall back to weighted ensemble of single-pathogen models |
| **Reporting delays** | High | Medium | Build nowcasting layer; use lagged features |
| **Concept drift** | Medium | Medium | Implement monitoring; retrain on rolling window |
| **Computational constraints** | Low | Low | Start with simpler models; scale as needed |

---

## 8. Success Criteria

### Minimum Viable Success
- [ ] Automated data pipeline pulling weekly NWSS + NHSN data
- [ ] Baseline model achieving MAE < 10 hospitalizations/100k at 1-week horizon
- [ ] Combined model outperforms sum of single-pathogen models by >5%

### Target Success
- [ ] 2-week forecast with MAE < 8 hospitalizations/100k
- [ ] Age-stratified predictions with balanced performance across groups
- [ ] Lead time quantification showing >7 days advance warning
- [ ] Publishable findings demonstrating novel multi-pathogen approach

### Stretch Goals
- [ ] Real-time dashboard for public health decision-makers
- [ ] Integration with CDC Forecast Hub evaluation framework
- [ ] Expansion to additional respiratory pathogens (HMPV, adenovirus)

---

## 9. References & Resources

### Data Sources
- CDC NWSS Main Page: https://www.cdc.gov/nwss/index.html
- CDC NWSS Data Documentation: https://www.cdc.gov/nwss/about-data.html
- WastewaterSCAN Dashboard: https://data.wastewaterscan.org/
- NHSN Hospital Respiratory Dashboard: https://www.cdc.gov/nhsn/psc/hospital-respiratory-dashboard.html
- HealthData.gov: https://healthdata.gov/

### Key Research Papers
1. "Wastewater-based epidemiology predicts COVID-19-induced weekly new hospital admissions" - Nature Communications
2. "COVID-19 Forecasting from U.S. Wastewater Surveillance Data Multi-Model Study" - arXiv 2512.01074
3. "Lead time of early warning by wastewater surveillance for COVID-19" - Journal of Hazardous Materials
4. "Wastewater surveillance provides 10-days forecasting of COVID-19 hospitalizations" - Infectious Disease Modelling
5. "Municipal and neighbourhood level wastewater surveillance of influenza" - Nature Scientific Reports
6. "Advancing health equity in wastewater-based epidemiology" - PMC 12005304

### Code Resources
- CDC CFA Wastewater-Informed Forecasting: https://github.com/CDCgov/wastewater-informed-covid-forecasting
- COVID-19 Forecast Hub: https://github.com/reichlab/covid19-forecast-hub

---

## Appendix A: Data Dictionary

### NWSS Public Metrics Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `wwtp_jurisdiction` | string | State/territory code |
| `wwtp_id` | string | Unique wastewater treatment plant identifier |
| `reporting_jurisdiction` | string | Reporting health department |
| `sample_collect_date` | date | Date sample was collected |
| `pcr_target` | string | Pathogen target (e.g., "sars-cov-2") |
| `percentile` | float | Current level as percentile of historical range |
| `ptc_15d` | float | Percent change over 15 days |
| `detect_prop_15d` | float | Detection proportion over 15 days |
| `population_served` | integer | Estimated population in sewershed |

### NHSN Hospital Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `jurisdiction` | string | State/territory |
| `week_end_date` | date | End date of reporting week |
| `totalconfirmed_covid_admissions` | integer | COVID-19 admissions |
| `totalconfirmed_flu_admissions` | integer | Influenza admissions |
| `totalconfirmed_rsv_admissions` | integer | RSV admissions |
| `inpatient_bed_covid` | integer | COVID patients in inpatient beds |
| `icu_covid` | integer | COVID patients in ICU |

---

## Appendix B: API Examples

### Fetching NWSS Data (Python)

```python
from sodapy import Socrata

# Initialize client (no auth needed for public data)
client = Socrata("data.cdc.gov", None)

# Fetch SARS-CoV-2 wastewater metrics
results = client.get(
    "2ew6-ywp6",
    limit=50000,
    where="sample_collect_date >= '2024-01-01'"
)

import pandas as pd
df = pd.DataFrame.from_records(results)
```

### Fetching NHSN Hospital Data (Python)

```python
import requests
import pandas as pd

url = "https://healthdata.gov/api/views/n3kj-exp9/rows.csv"
df = pd.read_csv(url)
```

---

*Report prepared for predictive modeling project scoping. All data sources are publicly available as of January 2026.*
