"""Pytest fixtures for wastewater disease prediction tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def sample_wastewater_data():
    """Generate sample wastewater surveillance data."""
    np.random.seed(42)  # Fixed seed for reproducibility
    dates = pd.date_range(start="2024-01-01", periods=20, freq="W")
    states = ["CA", "TX", "NY"]

    data = []
    for state in states:
        for date in dates:
            data.append(
                {
                    "state": state,
                    "week_end_date": date,
                    "percentile": np.random.uniform(20, 80),
                    "ptc_15d": np.random.uniform(-20, 20),
                    "population_served": np.random.randint(100000, 1000000),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_hospital_data():
    """Generate sample hospitalization data."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="W")
    states = ["CA", "TX", "NY"]

    data = []
    for state in states:
        for date in dates:
            data.append(
                {
                    "state": state,
                    "week_end_date": date,
                    "covid_hosp": np.random.randint(50, 500),
                    "flu_hosp": np.random.randint(20, 200),
                    "rsv_hosp": np.random.randint(10, 100),
                }
            )

    df = pd.DataFrame(data)
    df["respiratory_total"] = df["covid_hosp"] + df["flu_hosp"] + df["rsv_hosp"]
    return df


@pytest.fixture
def sample_merged_data(sample_wastewater_data, sample_hospital_data):
    """Generate merged wastewater + hospital data."""
    return pd.merge(
        sample_wastewater_data, sample_hospital_data, on=["state", "week_end_date"], how="inner"
    )


@pytest.fixture
def sample_county_data():
    """Generate sample county-level data."""
    return pd.DataFrame(
        {
            "fips": ["06001", "06002", "48001", "48002", "36001"],
            "county_name": ["Alameda", "Alpine", "Anderson", "Andrews", "Albany"],
            "state": ["California", "California", "Texas", "Texas", "New York"],
            "population": [1600000, 1200, 58000, 18000, 300000],
            "svi_overall": [0.45, 0.32, 0.67, 0.55, 0.41],
            "svi_quartile": ["Q2", "Q1 (Low)", "Q3", "Q3", "Q2"],
            "n_sites": [3, 0, 1, 0, 2],
            "coverage_pct": [75.0, 0.0, 45.0, 0.0, 60.0],
            "priority_score": [45.2, 78.5, 62.3, 71.0, 38.9],
            "priority_tier": ["Tier 2", "Tier 1 (Highest)", "Tier 2", "Tier 1 (Highest)", "Tier 3"],
        }
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    return tmp_path
