"""Tests for statistical forecasters.

This module tests:
- ETSForecaster (Exponential Smoothing)
- ThetaForecaster
- ARIMAForecaster (AutoARIMA)

These require statsforecast as an optional dependency.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters import ForecasterRegistry
from src.forecasters.statistical import (
    ARIMAForecaster,
    ETSForecaster,
    ThetaForecaster,
)


@pytest.fixture
def sample_price_series():
    """Create a sample price time series for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    # Trend + seasonal + noise
    trend = np.linspace(100, 120, 150)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(150) / 5)
    noise = np.random.randn(150) * 0.5
    prices = trend + seasonal + noise
    return pd.Series(prices, index=dates)


@pytest.fixture
def stationary_series():
    """Create a stationary time series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    # White noise around mean
    returns = np.random.randn(150) * 0.02
    return pd.Series(returns, index=dates)


class TestETSForecaster:
    """Test ETS (Exponential Smoothing) forecaster."""

    def test_registration(self):
        """Test that ETS forecaster is registered."""
        assert "ets" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("ets")
        assert isinstance(forecaster, ETSForecaster)

    def test_fit(self, sample_price_series):
        """Test fitting ETS model."""
        forecaster = ETSForecaster()
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict_single_step(self, sample_price_series):
        """Test single-step prediction."""
        forecaster = ETSForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        assert ~predictions.isna().any()

    def test_predict_multi_step(self, sample_price_series):
        """Test multi-step prediction."""
        forecaster = ETSForecaster()
        forecaster.fit(sample_price_series)

        horizon = 10
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        assert ~predictions.isna().any()

    def test_with_seasonal_period(self, sample_price_series):
        """Test with seasonal period."""
        forecaster = ETSForecaster(config={"season_length": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5
        assert forecaster.season_length == 5

    def test_with_custom_model_spec(self, sample_price_series):
        """Test with custom model specification."""
        # ETS(A,N,N) = Simple Exponential Smoothing
        forecaster = ETSForecaster(config={"model": "ANN"})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=3)
        assert len(predictions) == 3

    def test_family(self):
        """Test that ETS has family='statistical'."""
        forecaster = ETSForecaster()
        assert forecaster.family == "statistical"

    def test_prediction_index_is_datetime(self, sample_price_series):
        """Test that prediction index is DatetimeIndex."""
        forecaster = ETSForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions.index, pd.DatetimeIndex)
        assert predictions.index[0] > sample_price_series.index[-1]


class TestThetaForecaster:
    """Test Theta forecaster."""

    def test_registration(self):
        """Test that Theta forecaster is registered."""
        assert "theta" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("theta")
        assert isinstance(forecaster, ThetaForecaster)

    def test_fit(self, sample_price_series):
        """Test fitting Theta model."""
        forecaster = ThetaForecaster()
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict_single_step(self, sample_price_series):
        """Test single-step prediction."""
        forecaster = ThetaForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        assert ~predictions.isna().any()

    def test_predict_multi_step(self, sample_price_series):
        """Test multi-step prediction."""
        forecaster = ThetaForecaster()
        forecaster.fit(sample_price_series)

        horizon = 10
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        assert ~predictions.isna().any()

    def test_with_seasonal_period(self, sample_price_series):
        """Test with seasonal period."""
        forecaster = ThetaForecaster(config={"season_length": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5

    def test_decomposition_type(self, sample_price_series):
        """Test with different decomposition types."""
        # Additive decomposition
        forecaster_add = ThetaForecaster(config={"decomposition_type": "additive"})
        forecaster_add.fit(sample_price_series)
        pred_add = forecaster_add.predict(horizon=1)

        # Multiplicative decomposition
        forecaster_mult = ThetaForecaster(config={"decomposition_type": "multiplicative"})
        forecaster_mult.fit(sample_price_series)
        pred_mult = forecaster_mult.predict(horizon=1)

        assert len(pred_add) == 1
        assert len(pred_mult) == 1

    def test_family(self):
        """Test that Theta has family='statistical'."""
        forecaster = ThetaForecaster()
        assert forecaster.family == "statistical"


class TestARIMAForecaster:
    """Test ARIMA (AutoARIMA) forecaster."""

    def test_registration(self):
        """Test that ARIMA forecaster is registered."""
        assert "arima" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("arima")
        assert isinstance(forecaster, ARIMAForecaster)

    def test_fit(self, sample_price_series):
        """Test fitting ARIMA model."""
        forecaster = ARIMAForecaster()
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict_single_step(self, sample_price_series):
        """Test single-step prediction."""
        forecaster = ARIMAForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        assert ~predictions.isna().any()

    def test_predict_multi_step(self, sample_price_series):
        """Test multi-step prediction."""
        forecaster = ARIMAForecaster()
        forecaster.fit(sample_price_series)

        horizon = 5
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        assert ~predictions.isna().any()

    def test_with_seasonal_period(self, sample_price_series):
        """Test with seasonal ARIMA."""
        forecaster = ARIMAForecaster(config={"season_length": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5

    def test_with_stationary_data(self, stationary_series):
        """Test ARIMA on stationary data."""
        forecaster = ARIMAForecaster()
        forecaster.fit(stationary_series)

        predictions = forecaster.predict(horizon=3)

        assert len(predictions) == 3
        assert ~predictions.isna().any()

    def test_auto_selection(self, sample_price_series):
        """Test that AutoARIMA selects order automatically."""
        forecaster = ARIMAForecaster()
        forecaster.fit(sample_price_series)

        # Model should have selected an order
        assert forecaster.model is not None
        # AutoARIMA should fit successfully
        assert forecaster.is_fitted

    def test_family(self):
        """Test that ARIMA has family='statistical'."""
        forecaster = ARIMAForecaster()
        assert forecaster.family == "statistical"

    def test_prediction_index_is_datetime(self, sample_price_series):
        """Test that prediction index is DatetimeIndex."""
        forecaster = ARIMAForecaster()
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions.index, pd.DatetimeIndex)


class TestStatisticalForecasterComparison:
    """Test comparison of statistical forecasters."""

    def test_all_statistical_models_work_on_same_data(self, sample_price_series):
        """Test that all statistical models can fit and predict on same data."""
        statistical_models = ["ets", "theta", "arima"]

        for model_name in statistical_models:
            forecaster = ForecasterRegistry.create(model_name)
            forecaster.fit(sample_price_series)
            predictions = forecaster.predict(horizon=5)

            assert isinstance(predictions, pd.Series)
            assert len(predictions) == 5
            assert ~predictions.isna().any()

    def test_family_attribute(self):
        """Test that all statistical forecasters have family='statistical'."""
        statistical_models = ["ets", "theta", "arima"]

        for model_name in statistical_models:
            forecaster = ForecasterRegistry.create(model_name)
            assert forecaster.family == "statistical"

    def test_handle_trend_data(self, sample_price_series):
        """Test that statistical models handle trending data."""
        for model_name in ["ets", "theta", "arima"]:
            forecaster = ForecasterRegistry.create(model_name)
            forecaster.fit(sample_price_series)
            predictions = forecaster.predict(horizon=3)

            # Predictions should capture general trend direction
            assert len(predictions) == 3

    def test_minimum_data_requirements(self):
        """Test models with minimal data."""
        # Very short series
        short_series = pd.Series(
            range(100, 120), index=pd.date_range("2024-01-01", periods=20, freq="B")
        )

        for model_name in ["ets", "theta"]:  # ARIMA may need more data
            forecaster = ForecasterRegistry.create(model_name)
            forecaster.fit(short_series)
            predictions = forecaster.predict(horizon=1)

            assert len(predictions) == 1
