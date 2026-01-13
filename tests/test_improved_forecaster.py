"""Tests for the improved forecaster module."""

import pytest
import numpy as np
import pandas as pd


class TestCreateEnhancedFeatures:
    """Tests for create_enhanced_features function."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=20, freq="W")

        # Create realistic time series with trend
        base = 100
        values = [base + i * 5 + np.random.randint(-10, 10) for i in range(20)]

        return pd.DataFrame(
            {
                "week_end_date": dates,
                "respiratory_total": values,
                "covid_ww_percentile": [50 + np.random.uniform(-10, 10) for _ in range(20)],
            }
        )

    def test_creates_lag_features(self, sample_time_series):
        """Test that lag features are created correctly."""
        from src.models.improved_forecaster import create_enhanced_features

        df = create_enhanced_features(sample_time_series, target="respiratory_total")

        assert "respiratory_total_lag1" in df.columns
        assert "respiratory_total_lag2" in df.columns
        assert "respiratory_total_lag3" in df.columns
        assert "respiratory_total_lag4" in df.columns

        # Verify lag1 is actually the previous value
        assert df["respiratory_total_lag1"].iloc[5] == df["respiratory_total"].iloc[4]

    def test_creates_rolling_features(self, sample_time_series):
        """Test that rolling statistics are created."""
        from src.models.improved_forecaster import create_enhanced_features

        df = create_enhanced_features(sample_time_series, target="respiratory_total")

        assert "respiratory_total_roll2" in df.columns
        assert "respiratory_total_roll4" in df.columns

    def test_creates_momentum_features(self, sample_time_series):
        """Test that momentum (rate of change) features are created."""
        from src.models.improved_forecaster import create_enhanced_features

        df = create_enhanced_features(sample_time_series, target="respiratory_total")

        assert "respiratory_total_change1" in df.columns
        assert "respiratory_total_pct_change1" in df.columns
        assert "respiratory_total_accel" in df.columns

    def test_creates_wastewater_momentum(self, sample_time_series):
        """Test that wastewater momentum features are created."""
        from src.models.improved_forecaster import create_enhanced_features

        df = create_enhanced_features(sample_time_series, target="respiratory_total")

        assert "ww_change1" in df.columns
        assert "ww_pct_change1" in df.columns

    def test_ratio_to_roll4_handles_zero(self):
        """Test that ratio feature handles zero values without producing inf."""
        from src.models.improved_forecaster import create_enhanced_features

        # Create data with zeros
        df = pd.DataFrame(
            {
                "week_end_date": pd.date_range(start="2024-01-01", periods=10, freq="W"),
                "respiratory_total": [0, 0, 0, 0, 100, 200, 300, 400, 500, 600],
            }
        )

        result = create_enhanced_features(df, target="respiratory_total")

        # Should not have inf values
        assert not np.isinf(result["respiratory_total_ratio_to_roll4"]).any()

    def test_ratio_clipped_to_prevent_outliers(self):
        """Test that extreme ratios are clipped."""
        from src.models.improved_forecaster import create_enhanced_features

        # Create data with extreme jump
        df = pd.DataFrame(
            {
                "week_end_date": pd.date_range(start="2024-01-01", periods=10, freq="W"),
                "respiratory_total": [1, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000],
            }
        )

        result = create_enhanced_features(df, target="respiratory_total")

        # Ratios should be clipped to [0.01, 100]
        ratios = result["respiratory_total_ratio_to_roll4"].dropna()
        assert ratios.max() <= 100
        assert ratios.min() >= 0.01

    def test_creates_seasonal_features(self, sample_time_series):
        """Test that seasonal features are created."""
        from src.models.improved_forecaster import create_enhanced_features

        df = create_enhanced_features(sample_time_series, target="respiratory_total")

        assert "week_of_year" in df.columns
        assert "month" in df.columns
        assert "is_winter" in df.columns

        # Verify week_of_year is in valid range
        assert df["week_of_year"].min() >= 1
        assert df["week_of_year"].max() <= 53


class TestGetFeatureCols:
    """Tests for get_feature_cols function."""

    def test_returns_available_features(self):
        """Test that only available features are returned."""
        from src.models.improved_forecaster import get_feature_cols

        df = pd.DataFrame(
            {
                "respiratory_total_lag1": [1, 2, 3],
                "respiratory_total_lag2": [1, 2, 3],
                "week_of_year": [1, 2, 3],
                "month": [1, 2, 3],
                "is_winter": [1, 0, 1],
                # Missing some features
            }
        )

        cols = get_feature_cols(df, target="respiratory_total")

        # Should only include columns that exist
        for col in cols:
            assert col in df.columns

    def test_includes_wastewater_features_if_available(self):
        """Test that wastewater features are included when present."""
        from src.models.improved_forecaster import get_feature_cols

        df = pd.DataFrame(
            {
                "respiratory_total_lag1": [1, 2, 3],
                "covid_ww_percentile": [50, 55, 60],
                "ww_change1": [1, 2, 3],
            }
        )

        cols = get_feature_cols(df, target="respiratory_total")

        assert "covid_ww_percentile" in cols
        assert "ww_change1" in cols


class TestCreateSampleWeights:
    """Tests for create_sample_weights function."""

    def test_weights_decay_exponentially(self):
        """Test that weights decay exponentially with older samples."""
        from src.models.improved_forecaster import create_sample_weights

        weights = create_sample_weights(100, decay_factor=0.98)

        # Most recent should have highest weight
        assert weights[-1] > weights[0]

        # Weights should decrease as we go back
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1]

    def test_weights_are_normalized(self):
        """Test that weights are normalized to mean of 1."""
        from src.models.improved_forecaster import create_sample_weights

        weights = create_sample_weights(100, decay_factor=0.98)

        # Mean should be approximately 1
        assert abs(np.mean(weights) - 1.0) < 0.01

    def test_weights_length_matches_input(self):
        """Test that weight array length matches requested size."""
        from src.models.improved_forecaster import create_sample_weights

        for n in [10, 50, 100, 500]:
            weights = create_sample_weights(n)
            assert len(weights) == n


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n) * 0.5
        return X, y

    def test_fit_trains_all_models(self, sample_training_data):
        """Test that fit trains all component models."""
        from src.models.improved_forecaster import EnsembleForecaster

        X, y = sample_training_data
        forecaster = EnsembleForecaster()
        forecaster.fit(X, y)

        # All models should be trained (stored in models dict)
        assert "xgb" in forecaster.models
        assert "gbm" in forecaster.models
        assert "rf" in forecaster.models
        assert "ridge" in forecaster.models
        assert forecaster.models["xgb"] is not None

    def test_predict_returns_array(self, sample_training_data):
        """Test that predict returns numpy array."""
        from src.models.improved_forecaster import EnsembleForecaster

        X, y = sample_training_data
        forecaster = EnsembleForecaster()
        forecaster.fit(X, y)

        predictions = forecaster.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_adaptive_weights_during_surge(self, sample_training_data):
        """Test that ensemble uses adaptive weights during surge periods."""
        from src.models.improved_forecaster import EnsembleForecaster

        X, y = sample_training_data
        forecaster = EnsembleForecaster()
        forecaster.fit(X, y)

        # Create data with high momentum (surge signal)
        X_surge = np.random.randn(10, 5)
        X_surge[:, 0] = 50  # High momentum feature

        predictions = forecaster.predict(X_surge, recent_momentum=25)  # >20% = surge

        # Should still return valid predictions
        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))
