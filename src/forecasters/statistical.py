"""
Statistical Forecasters using StatsForecast

This module implements classical statistical time series models:
- ETS (Error, Trend, Seasonality / Exponential Smoothing)
- Theta: Simple method that performed well in M3 competition
- ARIMA: AutoRegressive Integrated Moving Average
- AutoARIMA: Automatic ARIMA model selection

These models are:
1. Statistically grounded (decades of research)
2. Interpretable (understand components)
3. Fast to train (CPU-friendly)
4. Strong baselines for financial forecasting
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA as SFAutoARIMA
from statsforecast.models import AutoETS as SFAutoETS
from statsforecast.models import AutoTheta as SFAutoTheta

from .base import BaseForecaster
from .registry import register_forecaster

logger = logging.getLogger(__name__)


@register_forecaster("ets")
class ETSForecaster(BaseForecaster):
    """
    ETS (Error, Trend, Seasonality) forecaster using exponential smoothing.

    Also known as Exponential Smoothing State Space Models.
    Automatically selects the best combination of:
    - Error type: Additive (A) or Multiplicative (M)
    - Trend type: None (N), Additive (A), or Multiplicative (M) with/without damping
    - Seasonality type: None (N), Additive (A), or Multiplicative (M)

    Use cases:
    - Smooth time series with trend/seasonality
    - Robust forecasting with automatic model selection
    - Interpretable components

    For financial data:
    - Often selects ANN (additive error, no trend, no seasonality)
    - Can capture momentum/trend in prices

    Config:
        season_length: int - Seasonal period (default: 5 for daily data = weekly pattern)
        model: str - ETS model specification, e.g., "AAN", "AAA" (default: "ZZZ" = auto)
        damped: bool - Whether to use damped trend (default: None = auto)
    """

    family = "statistical"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.season_length = self.config.get(
            "season_length", 1
        )  # 1 = no seasonality for daily returns
        self.model_spec = self.config.get("model", "ZZZ")  # ZZZ = auto-select
        self.damped = self.config.get("damped", None)
        self.statsforecast_model = None

    @property
    def model(self):
        """Alias for statsforecast_model for compatibility."""
        return self.statsforecast_model

    def fit(self, train_data: pd.Series) -> "ETSForecaster":
        """
        Fit ETS model on training data.

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        # Create and fit StatsForecast ETS model
        self.statsforecast_model = SFAutoETS(
            season_length=self.season_length, model=self.model_spec, damped=self.damped
        )

        # StatsForecast expects specific format
        y = train_data.values.astype(np.float64)

        try:
            self.statsforecast_model.fit(y=y)
            self._last_train_data = train_data
            self.is_fitted = True
            logger.debug(
                f"Fitted {self.name} on {len(train_data)} observations "
                f"(season_length={self.season_length})"
            )
        except Exception as e:
            logger.error(f"Failed to fit ETS model: {e}")
            raise ValueError(f"ETS fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate ETS forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of predictions with exponentially smoothed values
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        # Generate predictions
        try:
            predictions_dict = self.statsforecast_model.predict(h=horizon)
            predictions_values = predictions_dict["mean"]
        except Exception as e:
            logger.error(f"ETS prediction failed: {e}")
            raise RuntimeError(f"ETS prediction failed: {e}") from e

        # Generate future dates
        last_date = self._last_train_data.index[-1]
        freq = pd.infer_freq(self._last_train_data.index)
        if freq is None:
            freq = "B"

        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        predictions = pd.Series(data=predictions_values, index=future_dates)

        logger.debug(f"Generated {horizon}-step ETS forecast")
        return predictions


@register_forecaster("theta")
class ThetaForecaster(BaseForecaster):
    """
    Theta forecaster.

    The Theta method is a simple but effective forecasting method that
    performed very well in the M3 forecasting competition.

    How it works:
    1. Decomposes series into trend and seasonal components
    2. Extrapolates trend using a "theta coefficient"
    3. Combines components for final forecast

    The method is computationally efficient and works well for a wide
    variety of time series, including financial data.

    Use cases:
    - Fast, reliable baseline
    - Good for series with trend
    - Proven performance in competitions

    Config:
        season_length: int - Seasonal period (default: 1 = no seasonality)
        decomposition_type: str - "additive" or "multiplicative" (default: "additive")
    """

    family = "statistical"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.season_length = self.config.get("season_length", 1)
        self.decomposition_type = self.config.get("decomposition_type", "additive")
        self.statsforecast_model = None

    @property
    def model(self):
        """Alias for statsforecast_model for compatibility."""
        return self.statsforecast_model

    def fit(self, train_data: pd.Series) -> "ThetaForecaster":
        """
        Fit Theta model on training data.

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        # Create and fit StatsForecast Theta model
        self.statsforecast_model = SFAutoTheta(
            season_length=self.season_length, decomposition_type=self.decomposition_type
        )

        y = train_data.values.astype(np.float64)

        try:
            self.statsforecast_model.fit(y=y)
            self._last_train_data = train_data
            self.is_fitted = True
            logger.debug(
                f"Fitted {self.name} on {len(train_data)} observations "
                f"(season_length={self.season_length}, decomposition={self.decomposition_type})"
            )
        except Exception as e:
            logger.error(f"Failed to fit Theta model: {e}")
            raise ValueError(f"Theta fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate Theta forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of predictions using Theta method
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        try:
            predictions_dict = self.statsforecast_model.predict(h=horizon)
            predictions_values = predictions_dict["mean"]
        except Exception as e:
            logger.error(f"Theta prediction failed: {e}")
            raise RuntimeError(f"Theta prediction failed: {e}") from e

        last_date = self._last_train_data.index[-1]
        freq = pd.infer_freq(self._last_train_data.index)
        if freq is None:
            freq = "B"

        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        predictions = pd.Series(data=predictions_values, index=future_dates)

        logger.debug(f"Generated {horizon}-step Theta forecast")
        return predictions


@register_forecaster("arima")
class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster with automatic
    model selection.

    ARIMA(p, d, q) models capture:
    - p: AutoRegressive terms (dependence on past values)
    - d: Differencing order (to make series stationary)
    - q: Moving Average terms (dependence on past errors)

    This implementation uses AutoARIMA which automatically selects the best
    (p, d, q) parameters using information criteria (AIC, BIC).

    Use cases:
    - Stationary or near-stationary series
    - Capturing autocorrelation in returns
    - Classical econometric forecasting

    For financial data:
    - Prices: often ARIMA(0,1,0) = random walk
    - Returns: may find weak autocorrelations (p, q > 0)

    Config:
        season_length: int - Seasonal period (default: 1 = non-seasonal)
        seasonal: bool - Whether to fit seasonal ARIMA (default: False)
        approximation: bool - Use approximation for speed (default: True)
        max_p: int - Maximum AR order to test (default: 5)
        max_q: int - Maximum MA order to test (default: 5)
        max_d: int - Maximum differencing order (default: 2)
    """

    family = "statistical"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.season_length = self.config.get("season_length", 1)
        self.seasonal = self.config.get("seasonal", False)
        self.approximation = self.config.get("approximation", True)
        self.max_p = self.config.get("max_p", 5)
        self.max_q = self.config.get("max_q", 5)
        self.max_d = self.config.get("max_d", 2)
        self.statsforecast_model = None

    @property
    def model(self):
        """Alias for statsforecast_model for compatibility."""
        return self.statsforecast_model

    def fit(self, train_data: pd.Series) -> "ARIMAForecaster":
        """
        Fit ARIMA model with automatic order selection.

        Args:
            train_data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        # Minimum data requirement for ARIMA
        min_observations = max(10, 2 * self.season_length)
        if len(train_data) < min_observations:
            raise ValueError(
                f"ARIMA requires at least {min_observations} observations, "
                f"got {len(train_data)}"
            )

        # Create AutoARIMA model
        self.statsforecast_model = SFAutoARIMA(
            season_length=self.season_length,
            seasonal=self.seasonal,
            approximation=self.approximation,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
        )

        y = train_data.values.astype(np.float64)

        try:
            self.statsforecast_model.fit(y=y)
            self._last_train_data = train_data
            self.is_fitted = True
            logger.debug(
                f"Fitted {self.name} on {len(train_data)} observations "
                f"(seasonal={self.seasonal})"
            )
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            # Fallback: try simpler model
            logger.warning("Attempting fallback with simpler ARIMA configuration...")
            try:
                self.statsforecast_model = SFAutoARIMA(
                    season_length=1, seasonal=False, approximation=True, max_p=3, max_q=3, max_d=2
                )
                self.statsforecast_model.fit(y=y)
                self._last_train_data = train_data
                self.is_fitted = True
                logger.debug("Fallback ARIMA fit successful")
            except Exception as e2:
                logger.error(f"Fallback ARIMA also failed: {e2}")
                raise ValueError(f"ARIMA fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate ARIMA forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of ARIMA predictions
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        try:
            predictions_dict = self.statsforecast_model.predict(h=horizon)
            predictions_values = predictions_dict["mean"]
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            raise RuntimeError(f"ARIMA prediction failed: {e}") from e

        last_date = self._last_train_data.index[-1]
        freq = pd.infer_freq(self._last_train_data.index)
        if freq is None:
            freq = "B"

        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        predictions = pd.Series(data=predictions_values, index=future_dates)

        logger.debug(f"Generated {horizon}-step ARIMA forecast")
        return predictions


__all__ = [
    "ETSForecaster",
    "ThetaForecaster",
    "ARIMAForecaster",
]
