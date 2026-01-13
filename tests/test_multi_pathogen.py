"""Tests for the multi-pathogen model module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


class TestStateAbbrevMapping:
    """Tests for STATE_ABBREV mapping."""

    def test_all_50_states_mapped(self):
        """Test that all 50 states are mapped."""
        from src.models.multi_pathogen import STATE_ABBREV

        # Should have at least 50 states + DC
        assert len(STATE_ABBREV) >= 51

    def test_common_states_mapped_correctly(self):
        """Test that common states are mapped correctly."""
        from src.models.multi_pathogen import STATE_ABBREV

        assert STATE_ABBREV["California"] == "CA"
        assert STATE_ABBREV["Texas"] == "TX"
        assert STATE_ABBREV["New York"] == "NY"
        assert STATE_ABBREV["Florida"] == "FL"
        assert STATE_ABBREV["Illinois"] == "IL"

    def test_dc_included(self):
        """Test that DC is included."""
        from src.models.multi_pathogen import STATE_ABBREV

        assert "District of Columbia" in STATE_ABBREV
        assert STATE_ABBREV["District of Columbia"] == "DC"

    def test_territories_included(self):
        """Test that US territories are included."""
        from src.models.multi_pathogen import STATE_ABBREV

        assert "Puerto Rico" in STATE_ABBREV
        assert STATE_ABBREV["Puerto Rico"] == "PR"


class TestGetLatestFile:
    """Tests for get_latest_file function."""

    def test_returns_most_recent_file(self, tmp_path):
        """Test that most recent file is returned."""
        from src.models.multi_pathogen import get_latest_file

        # Create files with different dates
        (tmp_path / "data_20240101.parquet").touch()
        (tmp_path / "data_20240115.parquet").touch()
        (tmp_path / "data_20240201.parquet").touch()

        result = get_latest_file(tmp_path, "data_*.parquet")

        assert result.name == "data_20240201.parquet"

    def test_raises_error_when_no_files(self, tmp_path):
        """Test that FileNotFoundError is raised when no files match."""
        from src.models.multi_pathogen import get_latest_file

        with pytest.raises(FileNotFoundError, match="No files matching"):
            get_latest_file(tmp_path, "nonexistent_*.parquet")

    def test_handles_single_file(self, tmp_path):
        """Test with only one matching file."""
        from src.models.multi_pathogen import get_latest_file

        (tmp_path / "single_file.parquet").touch()

        result = get_latest_file(tmp_path, "*.parquet")

        assert result.name == "single_file.parquet"


class TestCreateMultiPathogenFeatures:
    """Tests for create_multi_pathogen_features function."""

    @pytest.fixture
    def sample_merged_data(self):
        """Create sample merged data."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=20, freq="W")
        states = ["CA", "TX"]

        data = []
        for state in states:
            for date in dates:
                data.append(
                    {
                        "state": state,
                        "week_end_date": date,
                        "covid_hosp": np.random.randint(100, 500),
                        "flu_hosp": np.random.randint(50, 200),
                        "rsv_hosp": np.random.randint(20, 100),
                        "covid_ww_percentile": np.random.uniform(30, 70),
                        "flu_ww_conc": np.random.uniform(1000, 10000),
                    }
                )

        df = pd.DataFrame(data)
        df["respiratory_total"] = df["covid_hosp"] + df["flu_hosp"] + df["rsv_hosp"]
        return df

    def test_creates_lag_features(self, sample_merged_data):
        """Test that lag features are created."""
        from src.models.multi_pathogen import create_multi_pathogen_features

        df = create_multi_pathogen_features(sample_merged_data.copy())

        # Check COVID lag features
        assert "covid_hosp_lag1" in df.columns
        assert "covid_hosp_lag2" in df.columns

    def test_creates_wastewater_lag_features(self, sample_merged_data):
        """Test that wastewater lag features are created."""
        from src.models.multi_pathogen import create_multi_pathogen_features

        df = create_multi_pathogen_features(sample_merged_data.copy())

        assert "covid_ww_percentile_lag1" in df.columns

    def test_creates_respiratory_total_features(self, sample_merged_data):
        """Test that respiratory total features are created."""
        from src.models.multi_pathogen import create_multi_pathogen_features

        df = create_multi_pathogen_features(sample_merged_data.copy())

        assert "respiratory_total_lag1" in df.columns


class TestBuildXgboostModel:
    """Tests for build_xgboost_model function."""

    @pytest.fixture
    def sample_train_test_data(self):
        """Create sample train and test data with features."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="W")

        df = pd.DataFrame(
            {
                "week_end_date": dates,
                "state": ["CA"] * n,
                "covid_hosp": np.random.randint(100, 500, n),
                "covid_hosp_lag1": np.random.randint(100, 500, n),
                "covid_hosp_lag2": np.random.randint(100, 500, n),
                "covid_hosp_lag3": np.random.randint(100, 500, n),
                "covid_hosp_roll2": np.random.randint(100, 500, n),
                "covid_ww_percentile": np.random.uniform(30, 70, n),
                "covid_ww_percentile_lag1": np.random.uniform(30, 70, n),
                "week_of_year": [d.isocalendar().week for d in dates],
                "month": [d.month for d in dates],
            }
        )

        # Split into train/test
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        return train, test

    def test_builds_model_without_wastewater(self, sample_train_test_data):
        """Test that model can be built without wastewater features."""
        from src.models.multi_pathogen import build_xgboost_model

        train, test = sample_train_test_data

        # Remove wastewater columns
        train = train.drop(columns=["covid_ww_percentile", "covid_ww_percentile_lag1"])
        test = test.drop(columns=["covid_ww_percentile", "covid_ww_percentile_lag1"])

        # Should still build successfully
        results = build_xgboost_model(train, test, target_col="covid_hosp", use_covid_ww=False)

        assert "model" in results
        assert "mae" in results

    def test_builds_model_with_wastewater(self, sample_train_test_data):
        """Test that model can use wastewater features."""
        from src.models.multi_pathogen import build_xgboost_model

        train, test = sample_train_test_data

        results = build_xgboost_model(
            train, test, target_col="covid_hosp", use_covid_ww=True
        )

        assert "model" in results
        assert "mae" in results


class TestTrainTestSplitTemporal:
    """Tests for train_test_split_temporal function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="W")
        return pd.DataFrame(
            {"week_end_date": dates, "value": range(20), "state": ["CA"] * 20}
        )

    def test_split_maintains_temporal_order(self, sample_data):
        """Test that test set comes after train set."""
        from src.models.multi_pathogen import train_test_split_temporal

        train, test = train_test_split_temporal(sample_data, test_weeks=4)

        train_max = train["week_end_date"].max()
        test_min = test["week_end_date"].min()

        assert train_max < test_min

    def test_no_overlap_between_train_test(self, sample_data):
        """Test that there's no overlap between train and test."""
        from src.models.multi_pathogen import train_test_split_temporal

        train, test = train_test_split_temporal(sample_data, test_weeks=4)

        train_dates = set(train["week_end_date"])
        test_dates = set(test["week_end_date"])

        assert train_dates.isdisjoint(test_dates)
