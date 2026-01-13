"""Tests for health equity analysis modules."""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


class TestStateSviRankings:
    """Tests for state SVI data."""

    def test_svi_scores_in_valid_range(self):
        """Test that all SVI scores are between 0 and 1."""
        from src.analysis.health_equity_ratios import get_state_svi_rankings

        svi_df = get_state_svi_rankings()

        assert svi_df["svi_score"].min() >= 0
        assert svi_df["svi_score"].max() <= 1

    def test_all_states_have_svi(self):
        """Test that all 50 states plus DC have SVI scores."""
        from src.analysis.health_equity_ratios import get_state_svi_rankings

        svi_df = get_state_svi_rankings()

        # At minimum should have 50 states + DC
        assert len(svi_df) >= 51

    def test_svi_dataframe_structure(self):
        """Test that SVI dataframe has expected columns."""
        from src.analysis.health_equity_ratios import get_state_svi_rankings

        svi_df = get_state_svi_rankings()

        assert "state" in svi_df.columns
        assert "svi_score" in svi_df.columns


class TestStateNameMapping:
    """Tests for state name to abbreviation mapping."""

    def test_all_states_mapped(self):
        """Test that all state names map to abbreviations."""
        from src.analysis.health_equity_ratios import get_state_name_to_abbrev

        mapping = get_state_name_to_abbrev()

        assert len(mapping) >= 50
        assert mapping["California"] == "CA"
        assert mapping["Texas"] == "TX"
        assert mapping["New York"] == "NY"

    def test_dc_included(self):
        """Test that DC is included in mapping."""
        from src.analysis.health_equity_ratios import get_state_name_to_abbrev

        mapping = get_state_name_to_abbrev()

        assert "District of Columbia" in mapping
        assert mapping["District of Columbia"] == "DC"


class TestHospWwRatios:
    """Tests for hospitalization-to-wastewater ratio calculations."""

    def test_ratio_calculation_with_valid_data(self):
        """Test ratio calculation produces valid results."""
        from src.analysis.health_equity_ratios import calculate_hosp_ww_ratios

        # Create sample data
        df = pd.DataFrame({
            "state": ["CA"] * 20 + ["TX"] * 20,
            "week_end_date": list(pd.date_range("2024-01-01", periods=20, freq="W")) * 2,
            "ww_percentile": np.random.uniform(30, 70, 40),
            "covid_hosp": np.random.randint(100, 500, 40),
        })

        results = calculate_hosp_ww_ratios(df, lag_weeks=1)

        assert len(results) > 0
        assert "state" in results.columns
        assert "hosp_ww_r2" in results.columns

    def test_ratio_r2_in_valid_range(self):
        """Test that R² values are between 0 and 1."""
        from src.analysis.health_equity_ratios import calculate_hosp_ww_ratios

        # Create correlated data
        np.random.seed(42)
        ww = np.random.uniform(30, 70, 20)
        hosp = ww * 5 + np.random.normal(0, 10, 20)  # Correlated with noise

        df = pd.DataFrame({
            "state": ["CA"] * 20,
            "week_end_date": pd.date_range("2024-01-01", periods=20, freq="W"),
            "ww_percentile": ww,
            "covid_hosp": hosp,
        })

        results = calculate_hosp_ww_ratios(df, lag_weeks=1)

        if len(results) > 0:
            assert results["hosp_ww_r2"].iloc[0] >= 0
            assert results["hosp_ww_r2"].iloc[0] <= 1


class TestEquityPatterns:
    """Tests for equity pattern analysis."""

    def test_analyze_equity_patterns_structure(self):
        """Test that equity analysis returns expected structure."""
        from src.analysis.health_equity_ratios import analyze_equity_patterns

        # Create sample ratio data
        ratios_df = pd.DataFrame({
            "state": ["CA", "TX", "NY", "FL", "OH", "PA", "IL", "GA"],
            "hosp_ww_slope": [1.2, 1.5, 1.1, 1.3, 1.4, 1.0, 1.2, 1.6],
            "hosp_ww_r": [0.7, 0.8, 0.6, 0.75, 0.65, 0.55, 0.7, 0.85],
            "hosp_ww_r2": [0.49, 0.64, 0.36, 0.56, 0.42, 0.30, 0.49, 0.72],
            "hosp_ww_pvalue": [0.01, 0.001, 0.05, 0.01, 0.02, 0.1, 0.01, 0.001],
            "hosp_ww_mean_ratio": [5.0, 6.0, 4.5, 5.5, 5.2, 4.0, 5.0, 6.5],
            "hosp_ww_ratio_cv": [0.2, 0.15, 0.25, 0.18, 0.22, 0.3, 0.2, 0.12],
            "n_weeks": [18, 18, 18, 18, 18, 18, 18, 18],
            "mean_hosp": [200, 300, 250, 280, 220, 190, 230, 320],
            "mean_ww": [40, 50, 55, 50, 42, 47, 46, 49],
        })

        svi_df = pd.DataFrame({
            "state": ["CA", "TX", "NY", "FL", "OH", "PA", "IL", "GA"],
            "svi_score": [0.54, 0.58, 0.51, 0.55, 0.52, 0.47, 0.50, 0.59],
        })

        results = analyze_equity_patterns(ratios_df, svi_df)

        assert "data" in results
        assert "correlations" in results
        assert "quartile_comparison" in results
        assert "interpretation" in results

    def test_svi_quartile_creation(self):
        """Test that SVI quartiles are properly created."""
        from src.analysis.health_equity_ratios import analyze_equity_patterns

        ratios_df = pd.DataFrame({
            "state": ["S" + str(i) for i in range(12)],
            "hosp_ww_slope": np.random.uniform(0.5, 2.0, 12),
            "hosp_ww_r": np.random.uniform(0.3, 0.9, 12),
            "hosp_ww_r2": np.random.uniform(0.1, 0.8, 12),
            "hosp_ww_pvalue": np.random.uniform(0.001, 0.1, 12),
            "hosp_ww_mean_ratio": np.random.uniform(3, 8, 12),
            "hosp_ww_ratio_cv": np.random.uniform(0.1, 0.4, 12),
            "n_weeks": [18] * 12,
            "mean_hosp": np.random.randint(150, 350, 12),
            "mean_ww": np.random.uniform(35, 60, 12),
        })

        svi_df = pd.DataFrame({
            "state": ["S" + str(i) for i in range(12)],
            "svi_score": np.linspace(0.2, 0.8, 12),  # Evenly distributed
        })

        results = analyze_equity_patterns(ratios_df, svi_df)

        # Check quartiles were created
        assert "svi_quartile" in results["data"].columns
        quartiles = results["data"]["svi_quartile"].unique()
        assert len(quartiles) == 4
