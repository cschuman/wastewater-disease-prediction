"""Tests for the equity simulation module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


class TestGetLatestFile:
    """Tests for get_latest_file function."""

    def test_returns_most_recent_file(self, tmp_path):
        """Test that most recent file is returned."""
        from src.analysis.equity_simulation import get_latest_file

        # Create files with different dates (sorted alphabetically)
        (tmp_path / "data_20240101.parquet").touch()
        (tmp_path / "data_20240115.parquet").touch()
        (tmp_path / "data_20240201.parquet").touch()

        result = get_latest_file(tmp_path, "data_*.parquet")

        assert result.name == "data_20240201.parquet"

    def test_raises_error_when_no_files(self, tmp_path):
        """Test that FileNotFoundError is raised when no files match."""
        from src.analysis.equity_simulation import get_latest_file

        with pytest.raises(FileNotFoundError, match="No files matching"):
            get_latest_file(tmp_path, "nonexistent_*.parquet")

    def test_handles_single_file(self, tmp_path):
        """Test with only one matching file."""
        from src.analysis.equity_simulation import get_latest_file

        (tmp_path / "single_file.parquet").touch()

        result = get_latest_file(tmp_path, "*.parquet")

        assert result.name == "single_file.parquet"


class TestCostAssumptions:
    """Tests for cost assumption constants."""

    def test_setup_cost_is_reasonable(self):
        """Test that setup cost is within reasonable range."""
        from src.analysis.equity_simulation import COST_PER_SITE_SETUP

        # Should be between $50k and $200k based on CDC/EPA estimates
        assert 50000 <= COST_PER_SITE_SETUP <= 200000

    def test_annual_cost_is_reasonable(self):
        """Test that annual cost is within reasonable range."""
        from src.analysis.equity_simulation import COST_PER_SITE_ANNUAL

        # Should be between $30k and $100k per year
        assert 30000 <= COST_PER_SITE_ANNUAL <= 100000


class TestScenarioAEqualizePerCapita:
    """Tests for scenario_a_equalize_per_capita function."""

    @pytest.fixture
    def sample_baseline_data(self):
        """Create sample baseline data for scenario testing."""
        np.random.seed(42)

        # Create 100 sample counties
        data = {
            "fips": [f"{i:05d}" for i in range(100)],
            "state": ["CA"] * 25 + ["TX"] * 25 + ["NY"] * 25 + ["FL"] * 25,
            "county_name": [f"County_{i}" for i in range(100)],
            "population": np.random.randint(50000, 500000, 100),
            "svi_overall": np.random.uniform(0.1, 0.9, 100),
            "n_sites": np.random.randint(0, 5, 100),
            "pop_served": np.random.randint(0, 200000, 100),
            "sites_per_million": np.random.uniform(0, 10, 100),
            "coverage_pct": np.random.uniform(0, 100, 100),
            "has_monitoring": [True] * 50 + [False] * 50,
        }

        df = pd.DataFrame(data)
        df["svi_quartile"] = pd.qcut(
            df["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
        )
        return df

    def test_returns_dataframe_and_results(self, sample_baseline_data, capsys):
        """Test that function returns DataFrame and results dict."""
        from src.analysis.equity_simulation import scenario_a_equalize_per_capita

        df_result, results = scenario_a_equalize_per_capita(sample_baseline_data)

        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(results, dict)

    def test_results_contain_required_keys(self, sample_baseline_data, capsys):
        """Test that results contain required keys."""
        from src.analysis.equity_simulation import scenario_a_equalize_per_capita

        _, results = scenario_a_equalize_per_capita(sample_baseline_data)

        assert "scenario" in results
        assert "total_new_sites" in results
        assert "total_cost_setup" in results
        assert "total_cost_5yr" in results
        assert "target_rate" in results

    def test_new_sites_is_non_negative(self, sample_baseline_data, capsys):
        """Test that new sites needed is non-negative."""
        from src.analysis.equity_simulation import scenario_a_equalize_per_capita

        df_result, results = scenario_a_equalize_per_capita(sample_baseline_data)

        assert results["total_new_sites"] >= 0
        assert (df_result["new_sites_needed"] >= 0).all()


class TestScenarioBPrioritizeZeroCoverage:
    """Tests for scenario_b_prioritize_zero_coverage function."""

    @pytest.fixture
    def sample_baseline_data(self):
        """Create sample baseline data for scenario testing."""
        np.random.seed(42)

        data = {
            "fips": [f"{i:05d}" for i in range(100)],
            "state": ["CA"] * 25 + ["TX"] * 25 + ["NY"] * 25 + ["FL"] * 25,
            "county_name": [f"County_{i}" for i in range(100)],
            "population": np.random.randint(50000, 500000, 100),
            "svi_overall": np.random.uniform(0.1, 0.9, 100),
            "n_sites": np.random.randint(0, 5, 100),
            "pop_served": np.random.randint(0, 200000, 100),
            "sites_per_million": np.random.uniform(0, 10, 100),
            "coverage_pct": np.random.uniform(0, 100, 100),
            "has_monitoring": [True] * 50 + [False] * 50,
        }

        df = pd.DataFrame(data)
        df["svi_quartile"] = pd.qcut(
            df["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
        )
        return df

    def test_returns_dataframe_and_results(self, sample_baseline_data, capsys):
        """Test that function returns DataFrame and results dict."""
        from src.analysis.equity_simulation import scenario_b_prioritize_zero_coverage

        df_result, results = scenario_b_prioritize_zero_coverage(sample_baseline_data)

        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(results, dict)

    def test_results_contain_required_keys(self, sample_baseline_data, capsys):
        """Test that results contain required keys."""
        from src.analysis.equity_simulation import scenario_b_prioritize_zero_coverage

        _, results = scenario_b_prioritize_zero_coverage(sample_baseline_data)

        assert "scenario" in results
        assert "total_new_sites" in results
        assert "counties_zero_coverage" in results

    def test_priority_score_calculated(self, sample_baseline_data, capsys):
        """Test that priority score is calculated."""
        from src.analysis.equity_simulation import scenario_b_prioritize_zero_coverage

        df_result, _ = scenario_b_prioritize_zero_coverage(sample_baseline_data)

        assert "priority_score" in df_result.columns


class TestScenarioCMinimumThreshold:
    """Tests for scenario_c_minimum_threshold function."""

    @pytest.fixture
    def sample_baseline_data(self):
        """Create sample baseline data for scenario testing."""
        np.random.seed(42)

        data = {
            "fips": [f"{i:05d}" for i in range(100)],
            "state": ["CA"] * 25 + ["TX"] * 25 + ["NY"] * 25 + ["FL"] * 25,
            "county_name": [f"County_{i}" for i in range(100)],
            "population": np.random.randint(50000, 500000, 100),
            "svi_overall": np.random.uniform(0.1, 0.9, 100),
            "n_sites": np.random.randint(0, 5, 100),
            "pop_served": np.random.randint(0, 200000, 100),
            "sites_per_million": np.random.uniform(0, 10, 100),
            "coverage_pct": np.random.uniform(0, 100, 100),
            "has_monitoring": [True] * 50 + [False] * 50,
        }

        df = pd.DataFrame(data)
        df["svi_quartile"] = pd.qcut(
            df["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
        )
        return df

    def test_returns_dataframe_and_results(self, sample_baseline_data, capsys):
        """Test that function returns DataFrame and results dict."""
        from src.analysis.equity_simulation import scenario_c_minimum_threshold

        df_result, results = scenario_c_minimum_threshold(sample_baseline_data)

        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(results, dict)

    def test_default_threshold_is_50_percent(self, sample_baseline_data, capsys):
        """Test that default threshold is 50%."""
        from src.analysis.equity_simulation import scenario_c_minimum_threshold

        _, results = scenario_c_minimum_threshold(sample_baseline_data)

        assert results["min_coverage_pct"] == 50.0

    def test_custom_threshold_is_used(self, sample_baseline_data, capsys):
        """Test that custom threshold is used."""
        from src.analysis.equity_simulation import scenario_c_minimum_threshold

        _, results = scenario_c_minimum_threshold(sample_baseline_data, min_coverage_pct=75.0)

        assert results["min_coverage_pct"] == 75.0

    def test_higher_threshold_requires_more_sites(self, sample_baseline_data, capsys):
        """Test that higher threshold requires more sites."""
        from src.analysis.equity_simulation import scenario_c_minimum_threshold

        _, results_50 = scenario_c_minimum_threshold(sample_baseline_data, min_coverage_pct=50.0)
        _, results_75 = scenario_c_minimum_threshold(sample_baseline_data, min_coverage_pct=75.0)

        assert results_75["total_new_sites"] >= results_50["total_new_sites"]


class TestCalculateCurrentCoverageBySvi:
    """Tests for calculate_current_coverage_by_svi function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)

        data = {
            "fips": [f"{i:05d}" for i in range(40)],
            "state": ["CA"] * 10 + ["TX"] * 10 + ["NY"] * 10 + ["FL"] * 10,
            "county_name": [f"County_{i}" for i in range(40)],
            "population": np.random.randint(50000, 500000, 40),
            "svi_overall": list(np.linspace(0.1, 0.3, 10)) + list(np.linspace(0.3, 0.5, 10)) +
                          list(np.linspace(0.5, 0.7, 10)) + list(np.linspace(0.7, 0.9, 10)),
            "n_sites": np.random.randint(0, 5, 40),
            "pop_served": np.random.randint(0, 200000, 40),
            "sites_per_million": np.random.uniform(0, 10, 40),
            "coverage_pct": np.random.uniform(0, 100, 40),
            "has_monitoring": [True] * 20 + [False] * 20,
        }

        df = pd.DataFrame(data)
        df["svi_quartile"] = pd.qcut(
            df["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
        )
        return df

    def test_returns_dataframe(self, sample_data):
        """Test that function returns a DataFrame."""
        from src.analysis.equity_simulation import calculate_current_coverage_by_svi

        result = calculate_current_coverage_by_svi(sample_data)

        assert isinstance(result, pd.DataFrame)

    def test_has_four_svi_quartiles(self, sample_data):
        """Test that result has four SVI quartiles."""
        from src.analysis.equity_simulation import calculate_current_coverage_by_svi

        result = calculate_current_coverage_by_svi(sample_data)

        assert len(result) == 4

    def test_has_coverage_metrics(self, sample_data):
        """Test that result has coverage metrics."""
        from src.analysis.equity_simulation import calculate_current_coverage_by_svi

        result = calculate_current_coverage_by_svi(sample_data)

        assert "% Pop Covered" in result.columns or "Sites per Million" in result.columns


class TestCreatePriorityInvestmentMap:
    """Tests for create_priority_investment_map function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample baseline data."""
        np.random.seed(42)

        data = {
            "fips": [f"{i:05d}" for i in range(100)],
            "state": ["CA"] * 25 + ["TX"] * 25 + ["NY"] * 25 + ["FL"] * 25,
            "county_name": [f"County_{i}" for i in range(100)],
            "population": np.random.randint(50000, 500000, 100),
            "svi_overall": np.random.uniform(0.1, 0.9, 100),
            "n_sites": np.random.randint(0, 5, 100),
            "pop_served": np.random.randint(0, 200000, 100),
            "sites_per_million": np.random.uniform(0, 10, 100),
            "coverage_pct": np.random.uniform(0, 100, 100),
            "has_monitoring": [True] * 50 + [False] * 50,
        }

        df = pd.DataFrame(data)
        df["svi_quartile"] = pd.qcut(
            df["svi_overall"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
        )
        return df

    def test_returns_tuple(self, sample_data, capsys):
        """Test that function returns a tuple."""
        from src.analysis.equity_simulation import create_priority_investment_map

        result = create_priority_investment_map(sample_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_priority_tier_created(self, sample_data, capsys):
        """Test that priority tier is created."""
        from src.analysis.equity_simulation import create_priority_investment_map

        _, df_priority = create_priority_investment_map(sample_data, {})

        assert "priority_tier" in df_priority.columns

    def test_priority_score_calculated(self, sample_data, capsys):
        """Test that priority score is calculated."""
        from src.analysis.equity_simulation import create_priority_investment_map

        _, df_priority = create_priority_investment_map(sample_data, {})

        assert "priority_score" in df_priority.columns
        # All scores should be non-negative
        assert (df_priority["priority_score"] >= 0).all()
