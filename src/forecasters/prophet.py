"""
Prophet Forecaster
==================

Facebook Prophet implementation for time series forecasting.
Integrates with the forecaster registry pattern.

Prophet is particularly good at:
- Handling multiple seasonalities (daily, weekly, yearly)
- Incorporating holiday effects
- Robust to missing data
- Works well with time series with strong seasonal effects

Usage:
    >>> from src.forecasters import ForecasterRegistry
    >>> forecaster = ForecasterRegistry.create("prophet")
    >>> forecaster.fit(price_series)
    >>> predictions = forecaster.predict(horizon=5)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
from prophet import Prophet

from .base import BaseForecaster
from .registry import register_forecaster

logger = logging.getLogger(__name__)


def _normalise_holiday_name(name: str, holiday_name_map: dict[str, str]) -> str:
    """
    Convert calendar holiday names into Prophet-friendly labels.

    :param name: Original holiday name
    :param holiday_name_map: Mapping of holiday names to normalized names
    :return: Normalized holiday name
    """
    if mapped := holiday_name_map.get(name):
        return mapped
    cleaned = name.lower()
    for char in ("'", ",", ".", "\u2019"):
        cleaned = cleaned.replace(char, "")
    cleaned = cleaned.replace("&", "and").replace("-", "_")
    cleaned = "_".join(segment for segment in cleaned.split() if segment)
    return cleaned.strip("_")


def _get_us_trading_holidays(
    start_year: int = 2020,
    end_year: int = 2030,
    holiday_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch US trading holidays using the official exchange calendar.

    :param start_year: Start year for holiday list
    :param end_year: End year for holiday list
    :param holiday_name_map: Mapping for normalizing holiday names
    :return: DataFrame with columns: holiday, ds, lower_window, upper_window
    """
    if holiday_name_map is None:
        holiday_name_map = {}

    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    start = pd.Timestamp(date(start_year, 1, 1))
    end = pd.Timestamp(date(end_year, 12, 31))

    calendar = mcal.get_calendar("XNYS")
    holidays: list[dict[str, Any]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()

    if getattr(calendar, "regular_holidays", None) is not None:
        for rule in calendar.regular_holidays.rules:
            name = _normalise_holiday_name(rule.name, holiday_name_map)
            for holiday_date in rule.dates(start, end):
                timestamp = pd.Timestamp(holiday_date)
                if timestamp.tz is not None:
                    timestamp = timestamp.tz_localize(None)
                timestamp = timestamp.normalize()
                key = (name, timestamp)
                if key in seen:
                    continue
                seen.add(key)
                holidays.append({"holiday": name, "ds": timestamp})

    for holiday_date in getattr(calendar, "adhoc_holidays", []):
        timestamp = pd.Timestamp(holiday_date)
        if timestamp.tz is not None:
            timestamp = timestamp.tz_localize(None)
        timestamp = timestamp.normalize()
        if not (start <= timestamp <= end):
            continue
        key = ("adhoc_holiday", timestamp)
        if key in seen:
            continue
        seen.add(key)
        holidays.append({"holiday": "adhoc_holiday", "ds": timestamp})

    if not holidays:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    holidays_df = pd.DataFrame(holidays).drop_duplicates(subset=["holiday", "ds"])
    if holidays_df.empty:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
    holidays_df = holidays_df.sort_values("ds").reset_index(drop=True)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1
    holidays_df["upper_window"] = 1

    return holidays_df


@register_forecaster("prophet")
class ProphetForecaster(BaseForecaster):
    """
    Prophet forecaster using Facebook's Prophet library.

    Prophet is designed for forecasting time series data based on an additive model
    where non-linear trends are fit with yearly, weekly, and daily seasonality,
    plus holiday effects.

    :ivar family: "statistical" - Prophet is categorized as statistical
    :ivar yearly_seasonality: Whether to include yearly seasonality
    :ivar weekly_seasonality: Whether to include weekly seasonality
    :ivar daily_seasonality: Whether to include daily seasonality
    :ivar include_holidays: Whether to include US trading holidays

    Config Options:
        yearly_seasonality: bool - Enable yearly seasonality (default: True)
        weekly_seasonality: bool - Enable weekly seasonality (default: True)
        daily_seasonality: bool - Enable daily seasonality (default: False)
        include_holidays: bool - Include US trading holidays (default: True)
        holiday_name_map: dict - Custom holiday name mapping

    Example:
        >>> forecaster = ProphetForecaster(config={
        ...     "yearly_seasonality": True,
        ...     "weekly_seasonality": True,
        ...     "include_holidays": True
        ... })
        >>> forecaster.fit(price_series)
        >>> predictions = forecaster.predict(horizon=5)
    """

    family = "statistical"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Prophet forecaster.

        :param config: Configuration dictionary with Prophet parameters
        """
        super().__init__(config)
        self.yearly_seasonality = self.config.get("yearly_seasonality", True)
        self.weekly_seasonality = self.config.get("weekly_seasonality", True)
        self.daily_seasonality = self.config.get("daily_seasonality", False)
        self.include_holidays = self.config.get("include_holidays", True)
        self.holiday_name_map = self.config.get("holiday_name_map", {})
        self._prophet_model: Prophet | None = None

    @property
    def model(self) -> Prophet | None:
        """Return the underlying Prophet model."""
        return self._prophet_model

    def fit(self, train_data: pd.Series) -> ProphetForecaster:
        """
        Fit Prophet model on training data.

        :param train_data: Time series with DatetimeIndex
        :return: Self for method chaining
        :raises ValueError: If train_data is invalid
        """
        self._validate_train_data(train_data)

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({"ds": train_data.index, "y": train_data.values})

        # Prophet configuration
        prophet_params = {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
        }

        # Add holidays if enabled
        if self.include_holidays:
            start_date = train_data.index.min()
            end_date = train_data.index.max()

            if isinstance(start_date, date):
                start_year = start_date.year
                end_year = end_date.year
            else:
                start_year = pd.to_datetime(start_date).year
                end_year = pd.to_datetime(end_date).year

            # Get holidays and filter to relevant date range
            holidays = _get_us_trading_holidays(start_year - 1, end_year + 1, self.holiday_name_map)
            holidays = holidays[
                (holidays["ds"] >= pd.to_datetime(start_date))
                & (holidays["ds"] <= pd.to_datetime(end_date))
            ]

            if not holidays.empty:
                prophet_params["holidays"] = holidays
                logger.debug(f"Using {len(holidays)} trading holidays for Prophet model")

        # Create and fit Prophet model
        try:
            self._prophet_model = Prophet(**prophet_params)
            self._prophet_model.fit(df)
            self._last_train_data = train_data
            self.is_fitted = True
            logger.debug(f"Fitted Prophet on {len(train_data)} observations")
        except Exception as e:
            logger.error(f"Failed to fit Prophet model: {e}")
            raise ValueError(f"Prophet fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate Prophet forecasts.

        :param horizon: Number of steps ahead to forecast
        :return: Series of predictions with datetime index
        :raises RuntimeError: If model not fitted
        :raises ValueError: If horizon invalid
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        if self._prophet_model is None:
            raise RuntimeError("Prophet model is None despite being marked as fitted")

        # Get the last date from training data
        last_date = self._last_train_data.index[-1]

        # Infer frequency
        freq = pd.infer_freq(self._last_train_data.index)
        if freq is None:
            freq = "B"  # Business day

        # Create future dataframe
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        future = pd.DataFrame({"ds": future_dates})

        # Make prediction
        try:
            forecast = self._prophet_model.predict(future)
            predictions = pd.Series(
                data=forecast["yhat"].values,
                index=future_dates,
            )
            logger.debug(f"Generated {horizon}-step Prophet forecast")
            return predictions
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            raise RuntimeError(f"Prophet prediction failed: {e}") from e

    def predict_with_uncertainty(self, horizon: int = 1) -> pd.DataFrame:
        """
        Generate Prophet forecasts with uncertainty intervals.

        :param horizon: Number of steps ahead to forecast
        :return: DataFrame with columns: yhat, yhat_lower, yhat_upper
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        if self._prophet_model is None:
            raise RuntimeError("Prophet model is None")

        last_date = self._last_train_data.index[-1]
        freq = pd.infer_freq(self._last_train_data.index) or "B"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        future = pd.DataFrame({"ds": future_dates})

        forecast = self._prophet_model.predict(future)
        return pd.DataFrame(
            {
                "yhat": forecast["yhat"].values,
                "yhat_lower": forecast["yhat_lower"].values,
                "yhat_upper": forecast["yhat_upper"].values,
            },
            index=future_dates,
        )


__all__ = ["ProphetForecaster", "_get_us_trading_holidays", "_normalise_holiday_name"]
