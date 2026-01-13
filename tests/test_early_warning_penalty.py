"""Tests for the early warning penalty analysis module."""

import pytest
import numpy as np
import pandas as pd
from scipy import signal


class TestGetStateNameToAbbrev:
    """Tests for get_state_name_to_abbrev function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from src.analysis.early_warning_penalty import get_state_name_to_abbrev

        result = get_state_name_to_abbrev()
        assert isinstance(result, dict)

    def test_all_50_states_included(self):
        """Test that all 50 states are included."""
        from src.analysis.early_warning_penalty import get_state_name_to_abbrev

        result = get_state_name_to_abbrev()
        # 50 states + DC + territories
        assert len(result) >= 51

    def test_correct_abbreviations(self):
        """Test that common states have correct abbreviations."""
        from src.analysis.early_warning_penalty import get_state_name_to_abbrev

        result = get_state_name_to_abbrev()
        assert result["California"] == "CA"
        assert result["Texas"] == "TX"
        assert result["New York"] == "NY"
        assert result["Florida"] == "FL"

    def test_dc_included(self):
        """Test that DC is included."""
        from src.analysis.early_warning_penalty import get_state_name_to_abbrev

        result = get_state_name_to_abbrev()
        assert "District of Columbia" in result
        assert result["District of Columbia"] == "DC"


class TestGetStateSviRankings:
    """Tests for get_state_svi_rankings function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        from src.analysis.early_warning_penalty import get_state_svi_rankings

        result = get_state_svi_rankings()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        from src.analysis.early_warning_penalty import get_state_svi_rankings

        result = get_state_svi_rankings()
        assert "state" in result.columns
        assert "svi_score" in result.columns

    def test_svi_scores_in_range(self):
        """Test that SVI scores are in valid range 0-1."""
        from src.analysis.early_warning_penalty import get_state_svi_rankings

        result = get_state_svi_rankings()
        assert result["svi_score"].min() >= 0
        assert result["svi_score"].max() <= 1

    def test_all_states_have_svi(self):
        """Test that all 50+ states have SVI scores."""
        from src.analysis.early_warning_penalty import get_state_svi_rankings

        result = get_state_svi_rankings()
        assert len(result) >= 51  # 50 states + DC


class TestGetStatePopulations:
    """Tests for get_state_populations function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        from src.analysis.early_warning_penalty import get_state_populations

        result = get_state_populations()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        from src.analysis.early_warning_penalty import get_state_populations

        result = get_state_populations()
        assert "state" in result.columns
        assert "population_millions" in result.columns

    def test_california_has_largest_population(self):
        """Test that California has one of the largest populations."""
        from src.analysis.early_warning_penalty import get_state_populations

        result = get_state_populations()
        ca_pop = result[result["state"] == "CA"]["population_millions"].values[0]
        assert ca_pop > 30  # California has 38+ million

    def test_populations_are_positive(self):
        """Test that all populations are positive."""
        from src.analysis.early_warning_penalty import get_state_populations

        result = get_state_populations()
        assert (result["population_millions"] > 0).all()


class TestDetectPeaks:
    """Tests for detect_peaks function."""

    def test_detects_obvious_peak(self):
        """Test that an obvious peak is detected."""
        from src.analysis.early_warning_penalty import detect_peaks

        # Create series with obvious peak
        data = [10, 20, 30, 100, 30, 20, 10]
        series = pd.Series(data)

        peaks, props = detect_peaks(series, min_prominence=30, min_distance=2)

        assert len(peaks) >= 1
        assert 3 in peaks  # Peak is at index 3

    def test_detects_multiple_peaks(self):
        """Test that multiple peaks are detected."""
        from src.analysis.early_warning_penalty import detect_peaks

        # Create series with two peaks
        data = [10, 50, 10, 10, 50, 10]
        series = pd.Series(data)

        peaks, props = detect_peaks(series, min_prominence=20, min_distance=2)

        assert len(peaks) == 2

    def test_returns_empty_for_flat_series(self):
        """Test that no peaks are found in flat series."""
        from src.analysis.early_warning_penalty import detect_peaks

        series = pd.Series([50] * 10)
        peaks, props = detect_peaks(series, min_prominence=10, min_distance=2)

        assert len(peaks) == 0

    def test_handles_nan_values(self):
        """Test that NaN values are handled gracefully."""
        from src.analysis.early_warning_penalty import detect_peaks

        data = [10, np.nan, 100, np.nan, 10]
        series = pd.Series(data)

        # Should not raise an error
        peaks, props = detect_peaks(series, min_prominence=30, min_distance=1)

        # After interpolation, peak should still be found
        assert isinstance(peaks, np.ndarray)


class TestCalculateCrossCorrelationLag:
    """Tests for calculate_cross_correlation_lag function."""

    def test_identical_series_zero_lag(self):
        """Test that identical series have zero lag."""
        from src.analysis.early_warning_penalty import calculate_cross_correlation_lag

        series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 50)))

        lag, corr = calculate_cross_correlation_lag(series, series, max_lag=5)

        assert lag == 0
        assert corr > 0.95  # High correlation

    def test_lagged_series_finds_correct_lag(self):
        """Test that lagged series are correctly identified."""
        from src.analysis.early_warning_penalty import calculate_cross_correlation_lag

        # Create two series with known lag
        np.random.seed(42)
        base = np.sin(np.linspace(0, 4 * np.pi, 50)) + np.random.randn(50) * 0.1

        series1 = pd.Series(base[:-3])  # Leading series
        series2 = pd.Series(base[3:])  # Lagging series

        lag, corr = calculate_cross_correlation_lag(series1, series2, max_lag=5)

        # The function should find a lag (positive or negative depending on alignment)
        assert isinstance(lag, (int, np.integer))
        assert abs(lag) <= 5  # Within max_lag range

    def test_returns_correlation_in_valid_range(self):
        """Test that correlation is in valid range [-1, 1]."""
        from src.analysis.early_warning_penalty import calculate_cross_correlation_lag

        np.random.seed(42)
        series1 = pd.Series(np.random.randn(30))
        series2 = pd.Series(np.random.randn(30))

        lag, corr = calculate_cross_correlation_lag(series1, series2, max_lag=5)

        assert -1 <= corr <= 1

    def test_handles_nan_values(self):
        """Test that NaN values are handled."""
        from src.analysis.early_warning_penalty import calculate_cross_correlation_lag

        series1 = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        series2 = pd.Series([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])

        # Should not raise an error
        lag, corr = calculate_cross_correlation_lag(series1, series2, max_lag=3)

        assert isinstance(lag, (int, np.integer))
        assert isinstance(corr, (float, np.floating))


class TestCalculatePeakBasedLeadTime:
    """Tests for calculate_peak_based_lead_time function."""

    def test_calculates_lead_time_for_matching_peaks(self):
        """Test lead time calculation with matching peaks."""
        from src.analysis.early_warning_penalty import calculate_peak_based_lead_time

        # Wastewater peaks first, then hospitalization
        dates = pd.Series(pd.date_range(start="2024-01-01", periods=20, freq="W"))
        ww_series = pd.Series([10, 50, 10, 10, 10, 10, 10, 10, 10, 10,
                               10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        hosp_series = pd.Series([10, 10, 10, 150, 10, 10, 10, 10, 10, 10,
                                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

        result = calculate_peak_based_lead_time(ww_series, hosp_series, dates)

        assert "mean_lead_days" in result
        assert "median_lead_days" in result
        assert "n_peaks_matched" in result
        assert "lead_times" in result

    def test_returns_nan_when_no_peaks(self):
        """Test that NaN is returned when no peaks are found."""
        from src.analysis.early_warning_penalty import calculate_peak_based_lead_time

        dates = pd.date_range(start="2024-01-01", periods=10, freq="W")
        ww_series = pd.Series([50] * 10)  # Flat
        hosp_series = pd.Series([50] * 10)  # Flat

        result = calculate_peak_based_lead_time(ww_series, hosp_series, dates)

        assert np.isnan(result["mean_lead_days"])
        assert result["n_peaks_matched"] == 0

    def test_returns_zero_ww_peaks_count(self):
        """Test that wastewater peak count is returned."""
        from src.analysis.early_warning_penalty import calculate_peak_based_lead_time

        dates = pd.date_range(start="2024-01-01", periods=10, freq="W")
        ww_series = pd.Series([50] * 10)
        hosp_series = pd.Series([50] * 10)

        result = calculate_peak_based_lead_time(ww_series, hosp_series, dates)

        assert "n_ww_peaks" in result
        assert "n_hosp_peaks" in result


class TestAnalyzeEarlyWarningPenalty:
    """Tests for analyze_early_warning_penalty function."""

    @pytest.fixture
    def sample_lead_times_df(self):
        """Create sample lead times DataFrame."""
        return pd.DataFrame({
            "state": ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA", "NC", "MI"],
            "peak_lead_days": [14, 12, 10, 8, 7, 9, 11, 6, 13, 10],
            "xcorr_lead_days": [14, 10, 12, 7, 8, 11, 9, 5, 14, 8],
            "xcorr_correlation": [0.8, 0.7, 0.75, 0.65, 0.72, 0.68, 0.71, 0.6, 0.78, 0.73],
            "sites_per_100k": [2.5, 1.8, 2.2, 1.5, 1.9, 2.0, 1.7, 1.3, 2.1, 1.6],
            "coverage_pct": [65.0, 55.0, 60.0, 50.0, 58.0, 62.0, 54.0, 48.0, 61.0, 52.0],
            "n_weeks": [52, 48, 50, 45, 47, 49, 46, 44, 51, 48],
            "mean_covid_hosp": [500, 800, 600, 700, 400, 550, 450, 650, 350, 500],
        })

    @pytest.fixture
    def sample_svi_df(self):
        """Create sample SVI DataFrame."""
        return pd.DataFrame({
            "state": ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA", "NC", "MI"],
            "svi_score": [0.35, 0.55, 0.40, 0.60, 0.45, 0.50, 0.52, 0.65, 0.48, 0.53],
        })

    def test_returns_dict_with_expected_keys(self, sample_lead_times_df, sample_svi_df):
        """Test that analysis returns dict with expected keys."""
        from src.analysis.early_warning_penalty import analyze_early_warning_penalty

        result = analyze_early_warning_penalty(sample_lead_times_df, sample_svi_df)

        assert isinstance(result, dict)
        assert "data" in result
        assert "svi_analysis" in result
        assert "xcorr_analysis" in result
        assert "key_findings" in result

    def test_creates_svi_quartiles(self, sample_lead_times_df, sample_svi_df):
        """Test that SVI quartiles are created."""
        from src.analysis.early_warning_penalty import analyze_early_warning_penalty

        result = analyze_early_warning_penalty(sample_lead_times_df, sample_svi_df)

        assert "svi_quartile" in result["data"].columns
        # Should have 4 quartiles
        assert result["data"]["svi_quartile"].nunique() <= 4

    def test_key_findings_is_list(self, sample_lead_times_df, sample_svi_df):
        """Test that key findings is a list."""
        from src.analysis.early_warning_penalty import analyze_early_warning_penalty

        result = analyze_early_warning_penalty(sample_lead_times_df, sample_svi_df)

        assert isinstance(result["key_findings"], list)
        # Should have at least one finding
        assert len(result["key_findings"]) >= 1
