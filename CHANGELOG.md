# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Data fetching scripts for CDC NWSS and NHSN APIs
- Baseline forecasting models (Naive, Seasonal Naive, ARIMA, XGBoost)
- Health equity analysis modules
- SvelteKit web dashboard for visualization
- County-level priority scoring system
- Interactive map visualization with D3.js

### Security
- Input validation for date parameters to prevent SOQL injection
- Path validation to prevent directory traversal attacks
- URL whitelisting for external data sources
- Resource cleanup for API clients

## [0.1.0] - 2025-01-13

### Added
- Initial release
- Multi-pathogen respiratory disease forecasting
- CDC wastewater surveillance data integration
- NHSN hospitalization data integration
- XGBoost and ARIMA baseline models
- Equity simulation and analysis tools
- Static web dashboard with county explorer

[Unreleased]: https://github.com/cschuman/wastewater-disease-prediction/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cschuman/wastewater-disease-prediction/releases/tag/v0.1.0
