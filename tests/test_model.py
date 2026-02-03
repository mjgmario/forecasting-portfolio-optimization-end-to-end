"""Tests for Prophet forecaster module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_time_series():
    """Create sample time series for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
    return pd.Series(prices, index=dates)


class TestProphetForecaster:
    """Test Prophet forecaster."""

    def test_fit(self, sample_time_series) -> None:
        """Test fitting Prophet model."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster()
        forecaster.fit(sample_time_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_time_series) -> None:
        """Test predict method."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster()
        forecaster.fit(sample_time_series)
        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        predicted_price = float(predictions.iloc[0])
        assert predicted_price > 0

    def test_predict_point(self, sample_time_series) -> None:
        """Test predict_point method."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster()
        forecaster.fit(sample_time_series)
        predicted_price = forecaster.predict_point(horizon=1)

        assert isinstance(predicted_price, float)
        assert predicted_price > 0

    def test_predict_multiple_horizons(self, sample_time_series) -> None:
        """Test predict with multiple horizons."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster()
        forecaster.fit(sample_time_series)
        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5

    def test_get_us_trading_holidays(self) -> None:
        """Test US trading holidays generation."""
        from src.forecasters.prophet import _get_us_trading_holidays

        holidays = _get_us_trading_holidays(2024, 2024)

        assert isinstance(holidays, pd.DataFrame)
        assert len(holidays) > 0
        assert "holiday" in holidays.columns
        assert "ds" in holidays.columns
        assert "lower_window" in holidays.columns
        assert "upper_window" in holidays.columns

        # Check that all holidays have proper windows
        assert all(holidays["lower_window"] == -1)
        assert all(holidays["upper_window"] == 1)

        # Check date format
        assert pd.api.types.is_datetime64_any_dtype(holidays["ds"])

    def test_fit_with_holidays(self, sample_time_series) -> None:
        """Test that Prophet model includes holidays when fitting."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster(config={"include_holidays": True})
        forecaster.fit(sample_time_series)

        assert forecaster.model is not None
        # Check that holidays are included
        assert hasattr(forecaster.model, "holidays")
        if forecaster.model.holidays is not None:
            assert len(forecaster.model.holidays) > 0

    def test_fit_with_seasonality_config(self, sample_time_series) -> None:
        """Test that Prophet model has seasonality properly configured."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster(
            config={
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
            }
        )
        forecaster.fit(sample_time_series)

        assert forecaster.model is not None
        # Check seasonality settings
        assert forecaster.model.yearly_seasonality is True
        assert forecaster.model.weekly_seasonality is True
        assert forecaster.model.daily_seasonality is False

    def test_registry_registration(self, sample_time_series) -> None:
        """Test that Prophet is registered in the ForecasterRegistry."""
        from src.forecasters import ForecasterRegistry
        from src.forecasters.prophet import ProphetForecaster

        assert ForecasterRegistry.is_registered("prophet")
        forecaster = ForecasterRegistry.create("prophet")
        assert isinstance(forecaster, ProphetForecaster)
        assert forecaster.family == "statistical"

    def test_predict_with_uncertainty(self, sample_time_series) -> None:
        """Test predict_with_uncertainty method."""
        from src.forecasters.prophet import ProphetForecaster

        forecaster = ProphetForecaster()
        forecaster.fit(sample_time_series)
        forecast_df = forecaster.predict_with_uncertainty(horizon=5)

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 5
        assert "yhat" in forecast_df.columns
        assert "yhat_lower" in forecast_df.columns
        assert "yhat_upper" in forecast_df.columns
