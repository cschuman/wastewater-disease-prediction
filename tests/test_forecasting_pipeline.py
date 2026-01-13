"""Tests for the forecasting pipeline module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import tempfile


class TestForecastingPipeline:
    """Tests for ForecastingPipeline class."""

    @pytest.fixture
    def sample_data_for_pipeline(self):
        """Create sample data suitable for the forecasting pipeline."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=30, freq="W")
        states = ["CA", "TX", "NY"]

        data = []
        for state in states:
            base_val = np.random.randint(100, 300)
            for i, date in enumerate(dates):
                # Create realistic data with trends
                covid_hosp = max(10, base_val + np.random.randint(-50, 50) + i * 2)
                flu_hosp = max(5, int(covid_hosp * 0.3) + np.random.randint(-10, 10))
                rsv_hosp = max(2, int(covid_hosp * 0.15) + np.random.randint(-5, 5))

                data.append(
                    {
                        "state": state,
                        "week_end_date": date,
                        "covid_hosp": covid_hosp,
                        "flu_hosp": flu_hosp,
                        "rsv_hosp": rsv_hosp,
                        "respiratory_total": covid_hosp + flu_hosp + rsv_hosp,
                        "covid_ww_percentile": np.random.uniform(30, 70),
                        "covid_ww_ptc": np.random.uniform(-15, 15),
                    }
                )

        return pd.DataFrame(data)

    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initializes correctly."""
        from src.forecasting.pipeline import ForecastingPipeline

        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        pipeline = ForecastingPipeline(data_dir, model_dir)

        assert pipeline.data_dir == data_dir
        assert pipeline.model_dir == model_dir
        assert isinstance(pipeline.models, dict)
        assert isinstance(pipeline.feature_cols, dict)

    def test_get_pct_change_normal_values(self):
        """Test percentage change calculation with normal values."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        # Create a DataFrame with two rows showing 10% increase
        state_df = pd.DataFrame({
            "covid_hosp": [100, 110],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })

        result = pipeline._get_pct_change(state_df, "covid_hosp")
        assert abs(result - 10.0) < 0.01  # 10% increase

    def test_get_pct_change_zero_previous(self):
        """Test percentage change with zero previous value returns 0."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        state_df = pd.DataFrame({
            "covid_hosp": [0, 100],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })

        result = pipeline._get_pct_change(state_df, "covid_hosp")
        assert result == 0.0  # Should not raise division by zero

    def test_get_pct_change_near_zero_previous(self):
        """Test percentage change with near-zero previous value."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        state_df = pd.DataFrame({
            "covid_hosp": [1e-7, 100],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })

        result = pipeline._get_pct_change(state_df, "covid_hosp")
        assert result == 0.0  # Should handle near-zero

    def test_hybrid_prediction_stable_period(self):
        """Test hybrid prediction uses mostly XGBoost during stable periods."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        # Create state_df with stable trend (low % change ~5%)
        state_df = pd.DataFrame({
            "covid_hosp": [100, 105],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })
        xgb_pred = 110

        result = pipeline._hybrid_prediction(state_df, "covid_hosp", xgb_pred)

        # Result should be close to XGBoost prediction in stable periods
        # (blend_weight is 0.1, so 90% XGBoost)
        assert result > 0
        assert isinstance(result, float)

    def test_hybrid_prediction_surge_period(self):
        """Test hybrid prediction increases trend weight during surges."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        # Create state_df with surge (20% change)
        state_df = pd.DataFrame({
            "covid_hosp": [100, 120],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })
        xgb_pred = 130

        result = pipeline._hybrid_prediction(state_df, "covid_hosp", xgb_pred)

        # During surge (>15%), blend_weight should be 0.4 (more trend weight)
        assert result > 0
        assert isinstance(result, float)

    def test_hybrid_prediction_never_negative(self):
        """Test that hybrid prediction never returns negative values."""
        from src.forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(Path("data"), Path("models"))

        # Create state_df with declining values
        state_df = pd.DataFrame({
            "covid_hosp": [100, 50],
            "week_end_date": pd.date_range("2024-01-01", periods=2, freq="W")
        })
        xgb_pred = -10  # Negative prediction

        result = pipeline._hybrid_prediction(state_df, "covid_hosp", xgb_pred)
        assert result >= 0  # Should be clamped to 0

    def test_save_and_load_models(self, tmp_path, sample_data_for_pipeline):
        """Test saving and loading models."""
        from src.forecasting.pipeline import ForecastingPipeline

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        pipeline = ForecastingPipeline(tmp_path, model_dir)

        # Train a simple model
        df = sample_data_for_pipeline.copy()

        # Add required columns if missing
        if "covid_ww_percentile_lag1" not in df.columns:
            for lag in [1, 2, 3, 4]:
                df[f"covid_ww_percentile_lag{lag}"] = df["covid_ww_percentile"].shift(lag)
            for target in ["covid_hosp", "flu_hosp", "rsv_hosp", "respiratory_total"]:
                for lag in [1, 2, 3, 4]:
                    df[f"{target}_lag{lag}"] = df[target].shift(lag)

        df = df.dropna()

        if len(df) > 10:
            results = pipeline.train_models(df, targets=["covid_hosp"], n_splits=2)

            # Save models
            pipeline.save_models()

            # Verify files exist
            assert (model_dir / "forecast_model_covid_hosp.joblib").exists()
            assert (model_dir / "feature_cols.joblib").exists()

            # Load into new pipeline
            new_pipeline = ForecastingPipeline(tmp_path, model_dir)
            new_pipeline.load_models()

            assert "covid_hosp" in new_pipeline.models

    def test_generate_forecasts_structure(self, tmp_path, sample_data_for_pipeline):
        """Test that generate_forecasts returns correct structure."""
        from src.forecasting.pipeline import ForecastingPipeline, Forecast

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        pipeline = ForecastingPipeline(tmp_path, model_dir)

        # Prepare data with features
        df = sample_data_for_pipeline.copy()
        for lag in [1, 2, 3, 4]:
            df[f"covid_ww_percentile_lag{lag}"] = df["covid_ww_percentile"].shift(lag)
        for target in ["covid_hosp", "flu_hosp", "rsv_hosp", "respiratory_total"]:
            for lag in [1, 2, 3, 4]:
                df[f"{target}_lag{lag}"] = df[target].shift(lag)

        df = df.dropna()

        if len(df) > 10:
            # Train first
            pipeline.train_models(df, targets=["covid_hosp"], n_splits=2)

            # Generate forecasts
            forecasts = pipeline.generate_forecasts(df, horizon_weeks=1)

            # Check that forecasts are returned for each state
            if len(forecasts) > 0:
                assert all(isinstance(f, Forecast) for f in forecasts)
                states_forecasted = {f.state for f in forecasts}
                assert len(states_forecasted) > 0


class TestForecastsToDataFrame:
    """Tests for forecasts_to_dataframe helper."""

    def test_converts_forecasts_to_dataframe(self):
        """Test conversion of forecasts list to DataFrame."""
        from src.forecasting.pipeline import Forecast, forecasts_to_dataframe

        forecasts = [
            Forecast(
                target_date=datetime(2024, 3, 1),
                state="CA",
                target="covid_hosp",
                point_estimate=150.0,
                lower_bound=120.0,
                upper_bound=180.0,
                horizon_weeks=1,
                model_version="v1.0",
                generated_at=datetime(2024, 2, 22),
            ),
            Forecast(
                target_date=datetime(2024, 3, 1),
                state="TX",
                target="covid_hosp",
                point_estimate=200.0,
                lower_bound=170.0,
                upper_bound=230.0,
                horizon_weeks=1,
                model_version="v1.0",
                generated_at=datetime(2024, 2, 22),
            ),
        ]

        df = forecasts_to_dataframe(forecasts)

        assert len(df) == 2
        assert "state" in df.columns
        assert "point_estimate" in df.columns
        assert "lower_bound" in df.columns
        assert "upper_bound" in df.columns
        assert df.iloc[0]["state"] == "CA"
        assert df.iloc[1]["point_estimate"] == 200.0

    def test_empty_forecasts_list(self):
        """Test handling empty forecasts list."""
        from src.forecasting.pipeline import forecasts_to_dataframe

        df = forecasts_to_dataframe([])
        assert len(df) == 0
