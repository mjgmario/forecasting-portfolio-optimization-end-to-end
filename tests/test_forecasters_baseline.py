"""Tests for baseline forecasters.

This module tests:
- NaiveForecaster
- SeasonalNaiveForecaster
- DriftForecaster
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters import ForecasterRegistry
from src.forecasters.baselines import (
    DriftForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture
def sample_price_series():
    """Create a sample price time series for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")  # Business days
    # Random walk with drift
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
    return pd.Series(prices, index=dates)


@pytest.fixture
def seasonal_price_series():
    """Create a seasonal price time series for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Create series with weekly pattern (5 trading days)
    base = np.arange(100)
    seasonal = np.sin(2 * np.pi * np.arange(100) / 5) * 2
    prices = 100 + base * 0.1 + seasonal + np.random.randn(100) * 0.2
    return pd.Series(prices, index=dates)


class TestNaiveForecaster:
    """Test Naive (Random Walk) forecaster."""

    def test_registration(self):
        """Test that Naive forecaster is registered."""
        assert "naive" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("naive")
        assert isinstance(forecaster, NaiveForecaster)

    def test_fit(self, sample_price_series):
        """Test fitting Naive model."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster._last_train_data is not None
        assert len(forecaster._last_train_data) == len(sample_price_series)

    def test_predict_single_step(self, sample_price_series):
        """Test single-step prediction."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        # Naive: prediction equals last value
        assert predictions.iloc[0] == sample_price_series.iloc[-1]

    def test_predict_multi_step(self, sample_price_series):
        """Test multi-step prediction."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_price_series)

        horizon = 5
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        # All predictions equal last value
        assert all(predictions == sample_price_series.iloc[-1])

    def test_predict_before_fit_raises_error(self, sample_price_series):
        """Test that predicting before fitting raises error."""
        forecaster = NaiveForecaster()

        with pytest.raises(RuntimeError, match="not fitted|must be fitted"):
            forecaster.predict(horizon=1)

    def test_predict_index_is_datetime(self, sample_price_series):
        """Test that prediction index is DatetimeIndex."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions.index, pd.DatetimeIndex)
        # Predictions start after last training date
        assert predictions.index[0] > sample_price_series.index[-1]

    def test_with_config(self):
        """Test creating Naive forecaster with config."""
        config = {}  # Naive has no hyperparameters
        forecaster = ForecasterRegistry.create("naive", config=config)

        assert isinstance(forecaster, NaiveForecaster)
        assert forecaster.config == config


class TestSeasonalNaiveForecaster:
    """Test Seasonal Naive forecaster."""

    def test_registration(self):
        """Test that Seasonal Naive forecaster is registered."""
        assert "seasonal_naive" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("seasonal_naive")
        assert isinstance(forecaster, SeasonalNaiveForecaster)

    def test_fit(self, seasonal_price_series):
        """Test fitting Seasonal Naive model."""
        forecaster = SeasonalNaiveForecaster(config={"season_length": 5})
        forecaster.fit(seasonal_price_series)

        assert forecaster.is_fitted
        assert forecaster.season_length == 5

    def test_predict_repeats_pattern(self, seasonal_price_series):
        """Test that predictions repeat seasonal pattern."""
        season_length = 5
        forecaster = SeasonalNaiveForecaster(config={"season_length": season_length})
        forecaster.fit(seasonal_price_series)

        horizon = 10
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon

        # Check that pattern repeats
        train_values = seasonal_price_series.values
        for h in range(horizon):
            seasonal_idx = -(season_length - (h % season_length))
            if seasonal_idx == 0:
                seasonal_idx = -season_length
            expected_value = train_values[seasonal_idx]
            assert predictions.iloc[h] == expected_value

    def test_default_season_length(self, seasonal_price_series):
        """Test default season length (5 trading days)."""
        forecaster = SeasonalNaiveForecaster()
        forecaster.fit(seasonal_price_series)

        assert forecaster.season_length == 5

    def test_custom_season_length(self, seasonal_price_series):
        """Test custom season length."""
        forecaster = SeasonalNaiveForecaster(config={"season_length": 10})
        forecaster.fit(seasonal_price_series)

        assert forecaster.season_length == 10

    def test_warning_short_data(self):
        """Test warning when data length < season_length."""
        short_series = pd.Series(
            [100, 101, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )

        forecaster = SeasonalNaiveForecaster(config={"season_length": 5})

        with pytest.warns(UserWarning, match="Training data length"):
            forecaster.fit(short_series)


class TestDriftForecaster:
    """Test Drift (Random Walk with Drift) forecaster."""

    def test_registration(self):
        """Test that Drift forecaster is registered."""
        assert "drift" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("drift")
        assert isinstance(forecaster, DriftForecaster)

    def test_fit(self, sample_price_series):
        """Test fitting Drift model."""
        forecaster = DriftForecaster()
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.drift_value is not None

    def test_drift_calculation(self):
        """Test that drift is calculated correctly."""
        # Simple series with known drift
        dates = pd.date_range("2024-01-01", periods=11, freq="B")
        prices = pd.Series(range(100, 111), index=dates)  # Linear: 100, 101, ..., 110

        forecaster = DriftForecaster()
        forecaster.fit(prices)

        # Drift should be (110 - 100) / (11 - 1) = 1.0
        assert forecaster.drift_value == pytest.approx(1.0)

    def test_predict_single_step(self):
        """Test single-step prediction with drift."""
        dates = pd.date_range("2024-01-01", periods=11, freq="B")
        prices = pd.Series(range(100, 111), index=dates)

        forecaster = DriftForecaster()
        forecaster.fit(prices)

        predictions = forecaster.predict(horizon=1)

        # Should be last value + drift = 110 + 1 = 111
        assert len(predictions) == 1
        assert predictions.iloc[0] == pytest.approx(111.0)

    def test_predict_multi_step(self):
        """Test multi-step prediction with drift."""
        dates = pd.date_range("2024-01-01", periods=11, freq="B")
        prices = pd.Series(range(100, 111), index=dates)

        forecaster = DriftForecaster()
        forecaster.fit(prices)

        horizon = 5
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon

        # Check linear extrapolation
        expected = [111.0, 112.0, 113.0, 114.0, 115.0]
        np.testing.assert_allclose(predictions.values, expected, rtol=1e-10)

    def test_fit_requires_two_observations(self):
        """Test that fitting requires at least 2 observations."""
        single_point = pd.Series([100], index=pd.date_range("2024-01-01", periods=1))

        forecaster = DriftForecaster()

        with pytest.raises(ValueError, match="at least 2 observations"):
            forecaster.fit(single_point)

    def test_negative_drift(self):
        """Test with negative drift."""
        dates = pd.date_range("2024-01-01", periods=11, freq="B")
        prices = pd.Series(range(110, 99, -1), index=dates)  # Decreasing

        forecaster = DriftForecaster()
        forecaster.fit(prices)

        assert forecaster.drift_value == pytest.approx(-1.0)

        predictions = forecaster.predict(horizon=3)
        expected = [99.0, 98.0, 97.0]
        np.testing.assert_allclose(predictions.values, expected, rtol=1e-10)


class TestBaselineComparison:
    """Test comparison of baseline models."""

    def test_all_baselines_work_on_same_data(self, sample_price_series):
        """Test that all baseline models can fit and predict on same data."""
        baseline_models = ["naive", "seasonal_naive", "drift"]

        for model_name in baseline_models:
            forecaster = ForecasterRegistry.create(model_name)
            forecaster.fit(sample_price_series)
            predictions = forecaster.predict(horizon=5)

            assert isinstance(predictions, pd.Series)
            assert len(predictions) == 5
            assert all(~predictions.isna())

    def test_family_attribute(self):
        """Test that all baselines have family='baseline'."""
        baseline_models = ["naive", "seasonal_naive", "drift"]

        for model_name in baseline_models:
            forecaster = ForecasterRegistry.create(model_name)
            assert forecaster.family == "baseline"
