# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-pathogen respiratory disease forecasting using CDC wastewater surveillance data to predict hospital admissions. The project has two main components:
1. **Python ML pipeline** - Data fetching, feature engineering, and forecasting models
2. **SvelteKit web dashboard** - Interactive visualization of county-level health equity analysis

## Common Commands

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Install with dev tools
pip install -e ".[dev]"

# Run tests
pytest

# Run single test file
pytest tests/test_specific.py -v

# Linting
ruff check src/ scripts/
black src/ scripts/ --check
```

### Data Pipeline
```bash
# Fetch wastewater data from CDC NWSS
python scripts/fetch_nwss_data.py

# Fetch hospital admission data from NHSN
python scripts/fetch_nhsn_data.py

# Run forecasting pipeline
python -m src.forecasting.pipeline --horizon 2

# Run baseline model comparison
python -m src.models.baseline
```

### Web Dashboard (SvelteKit)
```bash
cd web

# Install dependencies
npm install

# Development server
npm run dev

# Build static JSON data from Python outputs
npm run build:data

# Full build (data + site)
npm run build:full

# Type checking
npm run check
```

## Architecture

### Data Flow
1. `scripts/fetch_*.py` → Download raw data to `data/raw/`
2. `src/models/multi_pathogen.py` → Merge wastewater + hospital data
3. `src/forecasting/pipeline.py` → Train XGBoost models with time series CV
4. `src/analysis/*.py` → Equity analysis, simulations
5. `web/scripts/build-data.py` → Convert to JSON for web dashboard
6. `web/static/data/*.json` → Consumed by SvelteKit pages

### Key Python Modules
- `src/models/baseline.py` - ARIMA/XGBoost baseline models
- `src/models/improved_forecaster.py` - Enhanced forecasting with lagged features
- `src/forecasting/pipeline.py` - Production forecasting pipeline
- `src/analysis/health_equity_ratios.py` - SVI-based equity analysis
- `src/analysis/equity_simulation.py` - County priority scoring

### Web Dashboard Structure
- Uses SvelteKit with Svelte 5 runes (`$state`, `$derived`, `$props`)
- Static site generation with prerendered data
- D3.js for maps and charts
- Data loaded in `+layout.ts` and passed to all routes
- Routes: `/` (dashboard), `/map`, `/states`, `/counties`, `/scenarios`

## Key Patterns

### Python
- Use `Path` from pathlib for all file operations
- Data stored as parquet in `data/raw/` with date suffixes
- Use `get_latest_file(directory, pattern)` to find most recent data files
- Always split data temporally before creating lag/rolling features to prevent leakage
- Targets: `covid_hosp`, `flu_hosp`, `rsv_hosp`, `respiratory_total`

### Svelte 5
- Use `$derived.by(() => {...})` for complex derived values with logic
- Use `$derived(expression)` only for simple one-liner derivations
- Clean up D3 event listeners in action destroy functions to prevent memory leaks
- Types defined in `web/src/lib/types/`

## Data Sources
- **NWSS** (Socrata API): Wastewater pathogen concentrations - `data.cdc.gov` dataset `2ew6-ywp6`
- **NHSN**: Weekly respiratory hospitalizations by state
- **SVI**: CDC Social Vulnerability Index (embedded in state_svi dictionary)

## Configuration
- `config.yaml` - Data source endpoints, feature settings, model parameters
- `pyproject.toml` - Python dependencies and tool configuration (black, ruff, pytest)
