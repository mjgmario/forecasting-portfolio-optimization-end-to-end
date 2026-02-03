"""Tests for Deep Learning forecasters (N-BEATS, N-HiTS).

This module tests:
- NBeatsForecaster: Neural Basis Expansion Analysis
- NHiTSForecaster: Neural Hierarchical Interpolation

Note: These tests use minimal configurations to run quickly.
Full model training would take much longer.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters import ForecasterRegistry
from src.forecasters.deep_learning import NBeatsForecaster, NHiTSForecaster


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Create trending data with some noise
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    prices = trend + noise
    return pd.Series(prices, index=dates)


@pytest.fixture
def minimal_config():
    """Minimal config for fast testing."""
    return {
        "h": 3,
        "input_size": 10,
        "max_steps": 5,  # Very few epochs for fast testing
        "batch_size": 16,
        "windows_batch_size": 32,
        "stack_types": ["identity", "trend"],
        "n_blocks": [1, 1],
        "mlp_units": [[64, 64], [64, 64]],  # Smaller model
    }


@pytest.fixture
def minimal_nhits_config():
    """Minimal config for N-HiTS fast testing."""
    return {
        "h": 3,
        "input_size": 10,
        "max_steps": 5,
        "batch_size": 16,
        "windows_batch_size": 32,
        "n_pool_kernel_size": [2, 2, 1],
        "n_freq_downsample": [2, 2, 1],
    }


class TestNBeatsForecaster:
    """Test N-BEATS forecaster."""

    def test_registration(self):
        """Test that N-BEATS is registered."""
        assert "nbeats" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("nbeats")
        assert isinstance(forecaster, NBeatsForecaster)

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        forecaster = NBeatsForecaster()

        assert forecaster.h == 5
        assert forecaster.input_size == 25  # 5 * h
        assert forecaster.max_steps == 500
        assert forecaster.learning_rate == 0.001
        assert forecaster.stack_types == ["identity", "trend", "seasonality"]
        assert forecaster.n_blocks == [1, 1, 1]
        assert forecaster.family == "deep_learning"
        assert forecaster.is_fitted is False

    def test_initialization_custom_config(self, minimal_config):
        """Test initialization with custom configuration."""
        forecaster = NBeatsForecaster(config=minimal_config)

        assert forecaster.h == 3
        assert forecaster.input_size == 10
        assert forecaster.max_steps == 5
        assert forecaster.stack_types == ["identity", "trend"]
        assert forecaster.n_blocks == [1, 1]

    def test_fit(self, sample_prices, minimal_config):
        """Test fitting N-BEATS model."""
        forecaster = NBeatsForecaster(config=minimal_config)
        result = forecaster.fit(sample_prices)

        # Check fit returns self (for chaining)
        assert result is forecaster
        assert forecaster.is_fitted is True
        assert forecaster.nf is not None
        assert forecaster._train_df is not None
        assert forecaster._last_train_data is not None

    def test_predict(self, sample_prices, minimal_config):
        """Test N-BEATS predictions."""
        forecaster = NBeatsForecaster(config=minimal_config)
        forecaster.fit(sample_prices)

        predictions = forecaster.predict(horizon=3)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 3
        assert isinstance(predictions.index, pd.DatetimeIndex)
        # Predictions should start after training data
        assert predictions.index[0] > sample_prices.index[-1]
        # Check values are reasonable (not NaN or extreme)
        assert not predictions.isna().any()
        assert all(50 < p < 200 for p in predictions)  # Within reasonable range

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises error."""
        forecaster = NBeatsForecaster()

        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(horizon=5)

    def test_insufficient_data_raises_error(self, minimal_config):
        """Test that insufficient data raises error."""
        short_data = pd.Series(
            [100, 101, 102, 103, 104], index=pd.date_range("2024-01-01", periods=5, freq="D")
        )

        forecaster = NBeatsForecaster(config=minimal_config)

        # input_size=10, h=3 -> needs at least 13 samples
        with pytest.raises(ValueError, match="Need at least"):
            forecaster.fit(short_data)

    def test_horizon_mismatch_warning(self, sample_prices, minimal_config, caplog):
        """Test warning when horizon doesn't match training."""
        forecaster = NBeatsForecaster(config=minimal_config)
        forecaster.fit(sample_prices)

        # Predict with different horizon than trained
        forecaster.predict(horizon=2)  # trained with h=3

        # Should log a warning
        assert any("Consider retraining" in record.message for record in caplog.records)


class TestNHiTSForecaster:
    """Test N-HiTS forecaster."""

    def test_registration(self):
        """Test that N-HiTS is registered."""
        assert "nhits" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("nhits")
        assert isinstance(forecaster, NHiTSForecaster)

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        forecaster = NHiTSForecaster()

        assert forecaster.h == 5
        assert forecaster.input_size == 25
        assert forecaster.max_steps == 500
        assert forecaster.interpolation_mode == "linear"
        assert forecaster.family == "deep_learning"
        assert forecaster.is_fitted is False

    def test_initialization_custom_config(self, minimal_nhits_config):
        """Test initialization with custom configuration."""
        forecaster = NHiTSForecaster(config=minimal_nhits_config)

        assert forecaster.h == 3
        assert forecaster.input_size == 10
        assert forecaster.max_steps == 5
        assert forecaster.n_pool_kernel_size == [2, 2, 1]
        assert forecaster.n_freq_downsample == [2, 2, 1]

    def test_fit(self, sample_prices, minimal_nhits_config):
        """Test fitting N-HiTS model."""
        forecaster = NHiTSForecaster(config=minimal_nhits_config)
        result = forecaster.fit(sample_prices)

        assert result is forecaster
        assert forecaster.is_fitted is True
        assert forecaster.nf is not None
        assert forecaster._train_df is not None

    def test_predict(self, sample_prices, minimal_nhits_config):
        """Test N-HiTS predictions."""
        forecaster = NHiTSForecaster(config=minimal_nhits_config)
        forecaster.fit(sample_prices)

        predictions = forecaster.predict(horizon=3)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 3
        assert isinstance(predictions.index, pd.DatetimeIndex)
        assert predictions.index[0] > sample_prices.index[-1]
        assert not predictions.isna().any()
        assert all(50 < p < 200 for p in predictions)

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises error."""
        forecaster = NHiTSForecaster()

        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(horizon=5)

    def test_insufficient_data_raises_error(self, minimal_nhits_config):
        """Test that insufficient data raises error."""
        short_data = pd.Series(
            [100, 101, 102, 103, 104], index=pd.date_range("2024-01-01", periods=5, freq="D")
        )

        forecaster = NHiTSForecaster(config=minimal_nhits_config)

        with pytest.raises(ValueError, match="Need at least"):
            forecaster.fit(short_data)


class TestDeepLearningFamily:
    """Test deep learning forecaster family attributes."""

    def test_nbeats_family(self):
        """Test N-BEATS family attribute."""
        forecaster = NBeatsForecaster()
        assert forecaster.family == "deep_learning"

    def test_nhits_family(self):
        """Test N-HiTS family attribute."""
        forecaster = NHiTSForecaster()
        assert forecaster.family == "deep_learning"

    def test_list_by_family(self):
        """Test listing deep learning forecasters by family."""
        dl_forecasters = ForecasterRegistry.list_by_family("deep_learning")
        assert "nbeats" in dl_forecasters
        assert "nhits" in dl_forecasters


class TestDeepLearningRegistry:
    """Test registry operations for deep learning models."""

    def test_get_info_nbeats(self):
        """Test getting N-BEATS info from registry."""
        info = ForecasterRegistry.get_info("nbeats")
        assert info["name"] == "nbeats"
        assert info["class_name"] == "NBeatsForecaster"
        assert info["family"] == "deep_learning"
        assert "N-BEATS" in info["docstring"]

    def test_get_info_nhits(self):
        """Test getting N-HiTS info from registry."""
        info = ForecasterRegistry.get_info("nhits")
        assert info["name"] == "nhits"
        assert info["class_name"] == "NHiTSForecaster"
        assert info["family"] == "deep_learning"
        assert "N-HiTS" in info["docstring"]

    def test_create_with_config(self, minimal_config):
        """Test creating forecasters with config through registry."""
        forecaster = ForecasterRegistry.create("nbeats", config=minimal_config)
        assert forecaster.h == 3
        assert forecaster.max_steps == 5


class TestDeepLearningComparison:
    """Test comparing deep learning models."""

    def test_both_models_produce_output(self, sample_prices, minimal_config, minimal_nhits_config):
        """Test that both models produce valid predictions."""
        nbeats = NBeatsForecaster(config=minimal_config)
        nhits = NHiTSForecaster(config=minimal_nhits_config)

        nbeats.fit(sample_prices)
        nhits.fit(sample_prices)

        nbeats_pred = nbeats.predict(horizon=3)
        nhits_pred = nhits.predict(horizon=3)

        # Both should produce valid Series
        assert isinstance(nbeats_pred, pd.Series)
        assert isinstance(nhits_pred, pd.Series)
        assert len(nbeats_pred) == 3
        assert len(nhits_pred) == 3

        # Predictions should be similar but not identical
        # (models are different, but both should capture trend)
        assert not nbeats_pred.equals(nhits_pred)
