"""Tests for baseline forecasting models."""

import pytest
import numpy as np
import pandas as pd
from src.models.baseline import (
    ModelResults,
    evaluate_predictions,
    create_features,
    train_test_split_temporal,
    naive_baseline,
    seasonal_naive_baseline,
)


class TestModelResults:
    """Tests for ModelResults dataclass."""

    def test_model_results_creation(self):
        """Test creating a ModelResults instance."""
        results = ModelResults(
            model_name="test_model",
            mae=10.5,
            rmse=15.2,
            mape=12.3,
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 4]),
        )
        assert results.model_name == "test_model"
        assert results.mae == 10.5
        assert results.rmse == 15.2


class TestEvaluatePredictions:
    """Tests for prediction evaluation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        results = evaluate_predictions(y_true, y_pred, "perfect")

        assert results.mae == 0.0
        assert results.rmse == 0.0
        assert results.mape == 0.0

    def test_known_error(self):
        """Test metrics with known error values."""
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([110.0, 110.0, 110.0, 110.0])

        results = evaluate_predictions(y_true, y_pred, "known_error")

        assert results.mae == 10.0
        assert results.rmse == 10.0
        assert results.mape == 10.0  # 10% error

    def test_handles_nan_values(self):
        """Test that NaN values are properly filtered."""
        y_true = np.array([10.0, np.nan, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, np.nan, 40.0])

        results = evaluate_predictions(y_true, y_pred, "with_nan")

        # Should only use indices 0 and 3
        assert results.mae == 0.0
        assert len(results.actuals) == 2

    def test_mape_handles_zero_actuals(self):
        """Test that MAPE handles zero actual values."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        results = evaluate_predictions(y_true, y_pred, "zero_actuals")

        assert np.isnan(results.mape)


class TestCreateFeatures:
    """Tests for feature engineering."""

    def test_creates_lag_features(self, sample_merged_data):
        """Test that lag features are created."""
        df = create_features(sample_merged_data, target_col="covid_hosp")

        assert "covid_hosp_lag1" in df.columns
        assert "covid_hosp_lag2" in df.columns
        assert "covid_hosp_lag3" in df.columns
        assert "covid_hosp_lag4" in df.columns

    def test_creates_rolling_features(self, sample_merged_data):
        """Test that rolling features are created."""
        df = create_features(sample_merged_data, target_col="covid_hosp")

        assert "covid_hosp_roll2_mean" in df.columns
        assert "covid_hosp_roll4_mean" in df.columns

    def test_creates_time_features(self, sample_merged_data):
        """Test that time features are created."""
        df = create_features(sample_merged_data, target_col="covid_hosp")

        assert "week_of_year" in df.columns
        assert "month" in df.columns

    def test_handles_wastewater_features(self, sample_merged_data):
        """Test that wastewater lag features are created."""
        df = create_features(sample_merged_data, target_col="covid_hosp")

        assert "percentile_lag1" in df.columns
        assert "percentile_lag2" in df.columns


class TestTrainTestSplit:
    """Tests for temporal train/test split."""

    def test_split_maintains_temporal_order(self, sample_merged_data):
        """Test that test set comes after train set temporally."""
        train, test = train_test_split_temporal(sample_merged_data, test_weeks=4)

        train_max = train["week_end_date"].max()
        test_min = test["week_end_date"].min()

        assert train_max < test_min

    def test_split_sizes(self, sample_merged_data):
        """Test that split produces non-empty datasets."""
        train, test = train_test_split_temporal(sample_merged_data, test_weeks=4)

        assert len(train) > 0
        assert len(test) > 0

    def test_no_data_leakage(self, sample_merged_data):
        """Test that no test dates appear in training set."""
        train, test = train_test_split_temporal(sample_merged_data, test_weeks=4)

        train_dates = set(train["week_end_date"])
        test_dates = set(test["week_end_date"])

        assert train_dates.isdisjoint(test_dates)


class TestNaiveBaseline:
    """Tests for naive baseline model."""

    def test_naive_baseline_runs(self, sample_merged_data):
        """Test that naive baseline produces results."""
        df = create_features(sample_merged_data, target_col="covid_hosp")
        train, test = train_test_split_temporal(df, test_weeks=4)

        results = naive_baseline(train, test, "covid_hosp")

        assert results.model_name == "Naive (Last Value)"
        assert results.mae >= 0
        assert len(results.predictions) > 0

    def test_naive_uses_last_value(self):
        """Test that naive baseline uses last training value."""
        train = pd.DataFrame(
            {
                "state": ["CA"] * 5,
                "week_end_date": pd.date_range("2024-01-01", periods=5, freq="W"),
                "target": [10, 20, 30, 40, 50],
            }
        )
        test = pd.DataFrame(
            {
                "state": ["CA"] * 2,
                "week_end_date": pd.date_range("2024-02-05", periods=2, freq="W"),
                "target": [55, 60],
            }
        )

        results = naive_baseline(train, test, "target")

        # All predictions should be 50 (last training value)
        assert all(p == 50 for p in results.predictions)


class TestSeasonalNaiveBaseline:
    """Tests for seasonal naive baseline model."""

    def test_seasonal_naive_runs(self, sample_merged_data):
        """Test that seasonal naive baseline produces results."""
        df = create_features(sample_merged_data, target_col="covid_hosp")
        train, test = train_test_split_temporal(df, test_weeks=4)

        results = seasonal_naive_baseline(train, test, "covid_hosp")

        assert results.model_name == "Seasonal Naive"
        assert results.mae >= 0
        assert len(results.predictions) > 0
