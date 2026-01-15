# Roadmap

This document outlines the planned development direction for the Wastewater Disease Prediction project. It is a living document updated quarterly.

**Last Updated:** January 2025
**Status:** Active Development

---

## Vision

Build the most accurate and accessible open-source tool for predicting respiratory disease hospitalizations using wastewater surveillance data, enabling public health departments and researchers to prepare for disease surges 10-17 days in advance.

---

## Current Status: v0.1.x (Alpha)

The project is in **alpha**. Core functionality works but APIs may change.

### What Works
- Data fetching from CDC NWSS and NHSN APIs
- Baseline forecasting models (Naive, Seasonal, ARIMA, XGBoost)
- Health equity analysis with SVI integration
- Interactive web dashboard with county explorer
- 127 automated tests

### Known Limitations
- Single-state focus (national model in progress)
- Limited pathogen coverage (COVID, Flu, RSV only)
- No real-time prediction API
- Web dashboard requires manual data rebuilds

---

## Milestones

### v0.2.0 - Model Improvements (Q1 2025)
**Theme:** Better predictions, validated results

| Feature | Status | Priority |
|---------|--------|----------|
| Backtesting framework with historical validation | Planned | High |
| Improved feature engineering (weather, mobility) | Planned | High |
| Model comparison dashboard | Planned | Medium |
| Confidence intervals for all predictions | Planned | High |
| Documentation of model performance metrics | Planned | High |

**Success Criteria:**
- Backtested MAE < 8 hospitalizations/100k at 2-week horizon
- Published performance benchmarks

---

### v0.3.0 - Multi-State Expansion (Q2 2025)
**Theme:** National coverage

| Feature | Status | Priority |
|---------|--------|----------|
| All 50 states + territories support | Planned | High |
| Regional aggregation models | Planned | Medium |
| Cross-state correlation analysis | Planned | Medium |
| Automated data quality monitoring | Planned | High |

**Success Criteria:**
- Coverage of all states with NWSS data
- Data freshness < 7 days from CDC publication

---

### v0.4.0 - API & Integration (Q3 2025)
**Theme:** Enable downstream usage

| Feature | Status | Priority |
|---------|--------|----------|
| REST API for predictions | Planned | High |
| Webhook notifications for new forecasts | Planned | Medium |
| CSV/JSON export endpoints | Planned | High |
| Integration examples (Tableau, PowerBI) | Planned | Medium |

**Success Criteria:**
- API response time < 500ms
- 99.9% uptime for hosted version

---

### v1.0.0 - Production Ready (Q4 2025)
**Theme:** Stable, trusted, documented

| Feature | Status | Priority |
|---------|--------|----------|
| Semantic versioning guarantees | Planned | High |
| LTS support policy (12 months) | Planned | High |
| Complete API documentation | Planned | High |
| Security audit completed | Planned | High |
| OpenSSF Silver badge | Planned | Medium |

**Success Criteria:**
- Zero breaking changes without major version bump
- At least 3 documented production adopters
- Complete Diataxis documentation structure

---

## Future Exploration (2026+)

These are ideas under consideration, not committed:

| Idea | Notes |
|------|-------|
| Additional pathogens (Norovirus, Mpox) | Depends on data availability |
| International data sources | WHO, ECDC integration |
| Real-time streaming predictions | Infrastructure complexity |
| Mobile app | Community interest dependent |
| Ensemble with CDC FluSight models | Collaboration opportunity |

---

## Non-Goals

To maintain focus, we explicitly will **not**:

- Build clinical decision-support tools (we're research/planning focused)
- Provide individual-level predictions
- Replace official CDC forecasts
- Support proprietary data formats
- Build enterprise features behind paywalls

---

## How to Influence the Roadmap

### Request a Feature
1. Check existing [Issues](https://github.com/cschuman/wastewater-disease-prediction/issues) for duplicates
2. Open a new Issue with `[Feature Request]` prefix
3. Describe the use case, not just the solution
4. Community upvotes (reactions) help prioritize

### Propose a Change
1. For significant changes, open an `[RFC]` Issue
2. Include: motivation, proposed solution, alternatives considered
3. Allow 7 days for discussion
4. Maintainers will label with milestone if accepted

### Contribute Directly
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit PRs.

---

## Quarterly Review

This roadmap is reviewed and updated quarterly:

| Quarter | Review Date | Focus |
|---------|-------------|-------|
| Q1 2025 | April 1 | v0.2.0 retrospective |
| Q2 2025 | July 1 | v0.3.0 retrospective |
| Q3 2025 | October 1 | v0.4.0 retrospective |
| Q4 2025 | January 1 | v1.0.0 planning |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-14 | Initial roadmap published |

---

*This roadmap represents current intentions but is not a commitment. Priorities may shift based on community feedback, resource availability, and external factors.*
