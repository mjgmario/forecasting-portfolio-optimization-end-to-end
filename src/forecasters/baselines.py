"""
Baseline Forecasters.

This module implements classical baseline forecasting methods:
- Naive: Tomorrow = Today (Random Walk)
- Seasonal Naive: Repeat seasonal pattern
- Drift: Linear extrapolation of trend

These baselines are essential for:
1. Benchmarking more complex models
2. Computing MASE metrics
3. Understanding if complexity adds value
"""

import logging
import warnings
from typing import Any

import pandas as pd

from .base import BaseForecaster
from .registry import register_forecaster

logger = logging.getLogger(__name__)


@register_forecaster("naive")
class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster (Random Walk model).

    Forecast formula: ŷ(t+h) = y(t)

    This is the simplest forecaster and assumes tomorrow equals today.
    For financial time series (prices), this is remarkably strong due to
    the random walk nature of asset prices.

    Use cases:
    - Baseline for all forecasting comparisons
    - Computing MASE (Mean Absolute Scaled Error)
    - Quick sanity check

    Config:
        None - this model has no hyperparameters
    """

    family = "baseline"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def fit(self, train_data: pd.Series) -> "NaiveForecaster":
        """
        Fit Naive model (just stores last value).

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(f"Fitted {self.name} on {len(train_data)} observations")
        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate naive forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of predictions (all equal to last training value)
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        # Naive forecast: all future values equal the last observed value
        last_value = self._last_train_data.iloc[-1]
        last_date = self._last_train_data.index[-1]

        # Generate future dates using base class helper
        future_dates = self._make_future_index(last_date, horizon)

        predictions = pd.Series(data=[last_value] * horizon, index=future_dates)

        logger.debug(f"Generated {horizon}-step naive forecast: {last_value:.4f}")
        return predictions


@register_forecaster("seasonal_naive")
class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive forecaster.

    Forecast formula: ŷ(t+h) = y(t+h-m)
    where m is the seasonal period.

    This model repeats the seasonal pattern from the past.
    For financial data, daily seasonality is usually weak, but can be useful
    for intraday data or specific patterns (e.g., day-of-week effects).

    Use cases:
    - Baseline for seasonal data
    - Day-of-week or monthly effects
    - Comparing against seasonal patterns

    Config:
        season_length: int - Seasonal period (default: 5 for weekly pattern in daily data)
    """

    family = "baseline"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.season_length = self.config.get("season_length", 5)  # 5 trading days = 1 week

    def fit(self, train_data: pd.Series) -> "SeasonalNaiveForecaster":
        """
        Fit Seasonal Naive model.

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        if len(train_data) < self.season_length:
            msg = (
                f"Training data length ({len(train_data)}) < season_length ({self.season_length}). "
                "Predictions may be unreliable."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(
            f"Fitted {self.name} on {len(train_data)} observations "
            f"(season_length={self.season_length})"
        )
        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate seasonal naive forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of predictions following seasonal pattern
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        # Get the seasonal pattern
        train_values = self._last_train_data.values
        last_date = self._last_train_data.index[-1]
        n_train = len(train_values)

        # Use effective period (min of season_length and data length)
        effective_period = min(self.season_length, n_train)

        # Repeat the last effective_period values
        predictions_values = []
        for h in range(horizon):
            # Index from the end of training data, cycling through the period
            # h=0 -> index from position -effective_period + 0
            # h=1 -> index from position -effective_period + 1
            # etc., wrapping around
            offset = h % effective_period
            seasonal_idx = -effective_period + offset
            predictions_values.append(train_values[seasonal_idx])

        # Generate future dates using base class helper
        future_dates = self._make_future_index(last_date, horizon)

        predictions = pd.Series(data=predictions_values, index=future_dates)

        logger.debug(f"Generated {horizon}-step seasonal naive forecast")
        return predictions


@register_forecaster("drift")
class DriftForecaster(BaseForecaster):
    """
    Drift (Random Walk with Drift) forecaster.

    Forecast formula: ŷ(t+h) = y(t) + h * drift
    where drift = (y(t) - y(1)) / (t - 1)

    This model extrapolates a linear trend computed over the entire training window.
    It's equivalent to drawing a line from the first to the last observation
    and extending it forward.

    Use cases:
    - Baseline with trend
    - Understanding if linear extrapolation works
    - Comparing against trend-aware models

    Config:
        None - this model has no hyperparameters
    """

    family = "baseline"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.drift_value = None

    def fit(self, train_data: pd.Series) -> "DriftForecaster":
        """
        Fit Drift model.

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        if len(train_data) < 2:
            raise ValueError("Drift forecaster requires at least 2 observations")

        # Calculate drift: (last - first) / (n - 1)
        self.drift_value = (train_data.iloc[-1] - train_data.iloc[0]) / (len(train_data) - 1)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(
            f"Fitted {self.name} on {len(train_data)} observations "
            f"(drift={self.drift_value:.6f})"
        )
        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate drift forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of predictions with linear trend
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        last_value = self._last_train_data.iloc[-1]
        last_date = self._last_train_data.index[-1]

        # Forecast: last_value + h * drift for h = 1, 2, ..., horizon
        predictions_values = [last_value + h * self.drift_value for h in range(1, horizon + 1)]

        # Generate future dates using base class helper
        future_dates = self._make_future_index(last_date, horizon)

        predictions = pd.Series(data=predictions_values, index=future_dates)

        logger.debug(
            f"Generated {horizon}-step drift forecast " f"(starting at {predictions_values[0]:.4f})"
        )
        return predictions


# Export all baseline forecasters
__all__ = ["NaiveForecaster", "SeasonalNaiveForecaster", "DriftForecaster"]
