# Wastewater Disease Prediction

[![CI](https://github.com/cschuman/wastewater-disease-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/cschuman/wastewater-disease-prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cschuman/wastewater-disease-prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/cschuman/wastewater-disease-prediction)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Predictive modeling using CDC NWSS wastewater surveillance data to forecast respiratory disease hospitalizations before they manifest in clinical data.

## Project Goal

Build a **multi-pathogen combined respiratory burden forecasting model** that predicts total weekly hospitalizations (COVID-19 + Influenza + RSV) stratified by age group, using wastewater signals as leading indicators.

## Why This Matters

- Wastewater surveillance provides **10-17 days lead time** over clinical data
- Hospitals manage **total bed capacity**, not individual diseases
- No existing model predicts combined respiratory burden from multi-pathogen wastewater signals

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [CDC NWSS](https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6) | Wastewater pathogen concentrations | Public API |
| [NHSN Hospital Data](https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/n3kj-exp9) | Weekly respiratory hospitalizations | Public API |
| [WastewaterSCAN](https://data.wastewaterscan.org/) | Extended pathogen panel | Contact Stanford |

## Project Structure

```
wastewater-disease-prediction/
├── data/
│   ├── raw/           # Downloaded source data
│   ├── processed/     # Cleaned, feature-engineered data
│   └── external/      # Third-party data (weather, demographics)
├── notebooks/         # Exploratory analysis & experiments
├── src/
│   ├── data/          # Data fetching & processing
│   ├── features/      # Feature engineering
│   ├── models/        # Model implementations
│   └── evaluation/    # Metrics & validation
├── models/            # Trained model artifacts
├── reports/           # Analysis reports & documentation
├── scripts/           # CLI scripts for data pipeline
└── tests/             # Unit tests
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch data

```bash
# Fetch wastewater surveillance data
python scripts/fetch_nwss_data.py

# Fetch hospital admission data
python scripts/fetch_nhsn_data.py
```

### 3. Run exploratory analysis

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

## Prediction Target

**Weekly total respiratory hospitalizations by state and age group:**
- Pediatric (<18 years)
- Adult (18-64 years)
- Elderly (65+ years)

## Input Features

| Category | Features |
|----------|----------|
| Wastewater signals | SARS-CoV-2, Influenza A/B, RSV concentrations (normalized) |
| Temporal | Week of year, lagged values (1-4 weeks), trend indicators |
| Historical | Past hospitalization rates by age group |
| Population | Vaccination coverage, demographic composition |
| Environmental | Temperature, precipitation, humidity |
| Vulnerability | CDC Social Vulnerability Index |

## Modeling Approach

1. **Baseline:** ARIMA with wastewater covariates
2. **Ensemble:** XGBoost with temporal features
3. **Deep Learning:** LSTM/Transformer for sequence modeling
4. **Multi-task:** Shared encoder with age-group-specific heads

## Evaluation Metrics

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- WIS (Weighted Interval Score) for probabilistic forecasts
- Lead time quantification (days ahead of clinical detection)

## Success Criteria

- [ ] Automated weekly data pipeline
- [ ] Combined model outperforms sum of single-pathogen models by >5%
- [ ] 2-week forecast MAE < 8 hospitalizations/100k
- [ ] Demonstrated lead time >7 days

## Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Data Pipeline | Weeks 1-3 | API integration, validation |
| EDA & Features | Weeks 4-6 | Correlation analysis, feature engineering |
| Baseline Models | Weeks 7-10 | ARIMA, XGBoost benchmarks |
| Advanced Models | Weeks 11-16 | Neural architectures, tuning |
| Validation | Weeks 17-20 | Cross-validation, ablation studies |
| Documentation | Weeks 21-24 | Report, potential publication |

## Documentation

- [Full Research Report](reports/wastewater-disease-prediction-report.md) - Comprehensive project scoping
- [Roadmap](ROADMAP.md) - Project direction and planned features
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Governance](GOVERNANCE.md) - Decision-making process
- [Stability Policy](STABILITY.md) - API compatibility guarantees
- [Security Policy](SECURITY.md) - Reporting vulnerabilities

## Community

We welcome contributions! See our [Contributor Ladder](docs/CONTRIBUTOR_LADDER.md) for the path from first PR to maintainer.

- [GitHub Discussions](https://github.com/cschuman/wastewater-disease-prediction/discussions) - Questions and ideas
- [Issue Tracker](https://github.com/cschuman/wastewater-disease-prediction/issues) - Bugs and feature requests

## References

- CDC NWSS: https://www.cdc.gov/nwss/index.html
- CDC Wastewater Forecasting: https://github.com/CDCgov/wastewater-informed-covid-forecasting
- COVID-19 Forecast Hub: https://github.com/reichlab/covid19-forecast-hub
