"""
Base Forecaster Abstract Class
==============================

This module defines the abstract base class for all forecasting models.
All forecasters must inherit from BaseForecaster and implement the
:meth:`fit` and :meth:`predict` methods.

Example
-------
>>> from src.forecasters import ForecasterRegistry
>>> forecaster = ForecasterRegistry.create("xgboost", {"n_estimators": 200})
>>> forecaster.fit(train_data)
>>> predictions = forecaster.predict(horizon=5)
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.constants import FREQ_DEFAULT


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    All forecasting models must inherit from this class and implement
    the :meth:`fit` and :meth:`predict` methods.

    :ivar name: Human-readable name of the forecaster (auto-generated from class name)
    :vartype name: str
    :ivar family: Category of forecaster (baseline, statistical, volatility, ml, dl)
    :vartype family: str
    :ivar config: Configuration dictionary for model hyperparameters
    :vartype config: dict
    :ivar is_fitted: Whether the model has been fitted
    :vartype is_fitted: bool

    Example
    -------
    >>> class MyForecaster(BaseForecaster):
    ...     family = "custom"
    ...     def fit(self, train_data):
    ...         self._last_train_data = train_data
    ...         self.is_fitted = True
    ...         return self
    ...     def predict(self, horizon=1):
    ...         return pd.Series([self._last_train_data.iloc[-1]] * horizon)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the forecaster.

        :param config: Optional dictionary of model configuration/hyperparameters
        :type config: dict, optional
        """
        self.config = config or {}
        self.name = self.__class__.__name__.replace("Forecaster", "").lower()
        # Only set family if not already defined as class attribute
        if not hasattr(self.__class__, "family"):
            self.family = "unknown"
        self.is_fitted = False
        self._last_train_data: pd.Series | None = None

    @abstractmethod
    def fit(self, train_data: pd.Series) -> "BaseForecaster":
        """
        Fit the forecaster on training data.

        :param train_data: Time series with datetime index and numeric values
        :type train_data: pd.Series
        :return: Self for method chaining
        :rtype: BaseForecaster
        :raises ValueError: If train_data is invalid (empty, wrong type, etc.)
        """
        pass

    @abstractmethod
    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate forecasts for future time steps.

        :param horizon: Number of steps ahead to forecast
        :type horizon: int
        :return: Series of predictions with datetime index
        :rtype: pd.Series
        :raises RuntimeError: If model has not been fitted yet
        :raises ValueError: If horizon is invalid
        """
        pass

    def predict_point(self, horizon: int = 1) -> float:
        """
        Get single point forecast (convenience method).

        :param horizon: Number of steps ahead to forecast
        :type horizon: int
        :return: Single scalar prediction value
        :rtype: float
        :raises RuntimeError: If model has not been fitted yet
        """
        predictions = self.predict(horizon)
        return float(predictions.iloc[0])

    def get_config(self) -> dict[str, Any]:
        """
        Return model configuration for logging/serialization.

        :return: Dictionary of model hyperparameters and settings
        :rtype: dict
        """
        return {"name": self.name, "family": self.family, **self.config}

    @classmethod
    def get_name(cls) -> str:
        """
        Return human-readable model name.

        :return: Model name (e.g., "Prophet", "ARIMA", "XGBoost")
        :rtype: str
        """
        return cls.__name__.replace("Forecaster", "")

    @classmethod
    def get_family(cls) -> str:
        """
        Return forecaster family/category.

        :return: Family name (e.g., "baseline", "statistical", "ml", "dl")
        :rtype: str
        """
        # This will be overridden by the class-level family attribute
        return "unknown"

    def _validate_train_data(self, train_data: pd.Series) -> None:
        """
        Validate training data format and content.

        Checks:
        - Must be pandas Series with DatetimeIndex
        - Index must be sorted in ascending order
        - Index must have no duplicates
        - Must have at least one non-NaN value

        :param train_data: Time series to validate
        :type train_data: pd.Series
        :raises ValueError: If data is invalid
        """
        if not isinstance(train_data, pd.Series):
            raise ValueError(f"train_data must be pandas Series, got {type(train_data)}")

        if len(train_data) == 0:
            raise ValueError("train_data cannot be empty")

        if not isinstance(train_data.index, pd.DatetimeIndex):
            raise ValueError("train_data must have DatetimeIndex")

        # Check index is sorted
        if not train_data.index.is_monotonic_increasing:
            raise ValueError(
                "train_data index must be sorted in ascending order. "
                "Use train_data.sort_index() before fitting."
            )

        # Check for duplicate indices
        if train_data.index.has_duplicates:
            raise ValueError(
                "train_data index contains duplicate timestamps. "
                "Remove duplicates before fitting."
            )

        if train_data.isna().all():
            raise ValueError("train_data contains only NaN values")

    def _validate_fitted(self) -> None:
        """
        Check if model has been fitted.

        :raises RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions. "
                "Call fit() first."
            )

    def _validate_horizon(self, horizon: int) -> None:
        """
        Validate prediction horizon.

        :param horizon: Number of steps ahead
        :type horizon: int
        :raises ValueError: If horizon is invalid
        """
        if not isinstance(horizon, int):
            raise ValueError(f"horizon must be integer, got {type(horizon)}")

        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

    def _make_future_index(
        self,
        last_date: pd.Timestamp,
        horizon: int,
        freq: str | None = None,
    ) -> pd.DatetimeIndex:
        """
        Generate future date index for predictions.

        Infers frequency from training data if not provided.

        :param last_date: Last date in training data
        :type last_date: pd.Timestamp
        :param horizon: Number of future periods
        :type horizon: int
        :param freq: Frequency string (e.g., 'D', 'B', 'H'). If None, infers from train data.
        :type freq: str, optional
        :return: DatetimeIndex of future dates
        :rtype: pd.DatetimeIndex
        """
        # Infer frequency from training data if not provided
        if freq is None and self._last_train_data is not None:
            # Need at least 3 points to infer frequency reliably
            if len(self._last_train_data) >= 3:
                inferred = pd.infer_freq(self._last_train_data.index)
                freq = inferred if inferred else FREQ_DEFAULT
            else:
                freq = FREQ_DEFAULT
        elif freq is None:
            freq = FREQ_DEFAULT  # Default to business days for financial data

        return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    def __repr__(self) -> str:
        """String representation of forecaster."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(family={self.family}, {fitted_status})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.get_name()} Forecaster ({self.family})"
