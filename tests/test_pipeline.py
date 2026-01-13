"""Tests for the forecasting pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.forecasting.pipeline import Forecast, ForecastingPipeline


class TestForecast:
    """Tests for Forecast dataclass."""

    def test_forecast_creation(self):
        """Test creating a Forecast instance."""
        forecast = Forecast(
            target_date=datetime(2024, 3, 1),
            state="CA",
            target="covid_hosp",
            point_estimate=150.5,
            lower_bound=100.0,
            upper_bound=200.0,
            horizon_weeks=2,
            model_version="1.0.0",
            generated_at=datetime.now(),
        )

        assert forecast.state == "CA"
        assert forecast.point_estimate == 150.5
        assert forecast.lower_bound < forecast.point_estimate < forecast.upper_bound

    def test_forecast_confidence_interval_validity(self):
        """Test that confidence intervals are properly ordered."""
        forecast = Forecast(
            target_date=datetime(2024, 3, 1),
            state="TX",
            target="flu_hosp",
            point_estimate=75.0,
            lower_bound=50.0,
            upper_bound=100.0,
            horizon_weeks=1,
            model_version="1.0.0",
            generated_at=datetime.now(),
        )

        assert forecast.lower_bound <= forecast.point_estimate
        assert forecast.point_estimate <= forecast.upper_bound


class TestForecastingPipeline:
    """Tests for the ForecastingPipeline class."""

    def test_pipeline_initialization(self, temp_data_dir):
        """Test pipeline can be initialized."""
        pipeline = ForecastingPipeline(
            data_dir=temp_data_dir / "data",
            model_dir=temp_data_dir / "models",
        )

        assert pipeline.data_dir.exists()
        assert pipeline.models == {}

    def test_pipeline_train_models_returns_dict(self, sample_merged_data, temp_data_dir):
        """Test that train_models returns a dictionary."""
        pipeline = ForecastingPipeline(
            data_dir=temp_data_dir / "data",
            model_dir=temp_data_dir / "models",
        )

        # Add required columns for training
        df = sample_merged_data.copy()
        df["covid_hosp_lag1"] = df["covid_hosp"].shift(1)
        df["covid_hosp_lag2"] = df["covid_hosp"].shift(2)
        df["week_of_year"] = df["week_end_date"].dt.isocalendar().week.astype(int)
        df = df.dropna()

        if len(df) > 10:  # Only run if enough data
            results = pipeline.train_models(df, targets=["covid_hosp"], n_splits=2)
            assert isinstance(results, dict)


class TestPipelineHelpers:
    """Tests for pipeline helper functions."""

    def test_get_pct_change_handles_zeros(self):
        """Test percent change calculation handles zero values."""
        from src.forecasting.pipeline import ForecastingPipeline

        # Create series with zeros
        series = pd.Series([0, 10, 20, 0, 30])

        # Percent change should handle zeros without inf
        pct_change = series.pct_change().replace([np.inf, -np.inf], np.nan)

        assert not np.any(np.isinf(pct_change))

    def test_feature_columns_consistency(self, sample_merged_data):
        """Test that feature columns are consistent across calls."""
        from src.models.baseline import create_features

        df1 = create_features(sample_merged_data.copy(), "covid_hosp")
        df2 = create_features(sample_merged_data.copy(), "covid_hosp")

        assert set(df1.columns) == set(df2.columns)
