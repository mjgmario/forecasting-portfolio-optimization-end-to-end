"""
GARCH Volatility Forecasters

GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models
forecast volatility, not prices/returns.

Why GARCH for portfolio optimization?
1. Dynamic risk estimation: Σ(t) instead of static historical covariance
2. Volatility clustering: Models "calm" vs "turbulent" periods
3. Position sizing: Reduce exposure when volatility is high
4. Risk-adjusted returns: Better Sharpe ratios

GARCH models available:
- GARCH(1,1): Standard model (most common in finance)
- EGARCH: Exponential GARCH (handles leverage effect)
- GJR-GARCH: Threshold GARCH (asymmetric volatility response)

Output: Volatility forecast (standard deviation), not price/return forecast.
Usage in portfolio: Adjust covariance matrix or position sizes based on vol forecasts.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from .base import BaseForecaster
from .registry import register_forecaster

logger = logging.getLogger(__name__)


@register_forecaster("garch")
class GARCHForecaster(BaseForecaster):
    """
    GARCH(p, q) volatility forecaster.

    Standard GARCH model for forecasting conditional volatility.
    GARCH(1,1) is the most common specification in finance.

    Model: σ²(t) = ω + Σ α(i)*ε²(t-i) + Σ β(j)*σ²(t-j)

    Use cases:
    - Forecast volatility for risk management
    - Dynamic covariance matrix for portfolio optimization
    - Position sizing based on expected volatility
    - Volatility targeting strategies

    Important:
    - Input should be RETURNS, not prices
    - Output is volatility (standard deviation), not return
    - Typically fit on percentage returns (e.g., * 100)

    Config:
        p: int - GARCH order (lag of squared returns) (default: 1)
        q: int - ARCH order (lag of conditional variance) (default: 1)
        mean: str - Mean model ("Zero", "Constant", "AR") (default: "Constant")
        vol: str - Volatility model (default: "GARCH")
        dist: str - Error distribution ("normal", "t", "skewt") (default: "normal")
        rescale: bool - Rescale returns for numerical stability (default: True)
    """

    family = "volatility"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.p = self.config.get("p", 1)
        self.q = self.config.get("q", 1)
        self.mean = self.config.get("mean", "Constant")
        self.vol = self.config.get("vol", "GARCH")
        self.dist = self.config.get("dist", "normal")
        self.rescale = self.config.get("rescale", True)
        self.model: ARCHModelResult | None = None

    @property
    def fitted_model(self):
        """Alias for model for compatibility."""
        return self.model

    def fit(self, train_data: pd.Series) -> "GARCHForecaster":
        """
        Fit GARCH model on returns data.

        Args:
            train_data: Time series of RETURNS (not prices) with DatetimeIndex

        Returns:
            Self for method chaining
        """
        self._validate_train_data(train_data)

        # Minimum observations for GARCH - require enough data for numerical stability
        min_obs = max(30, 10 * (self.p + self.q))
        if len(train_data) < min_obs:
            raise ValueError(
                f"GARCH with p={self.p}, q={self.q} requires at least {min_obs} observations "
                f"for numerical stability. Got {len(train_data)}."
            )

        # Check if data looks like returns (should be centered around 0)
        if train_data.mean() > 0.5 or train_data.mean() < -0.5:
            logger.warning(
                "Input data has large mean. GARCH expects RETURNS (not prices). "
                f"Mean: {train_data.mean():.4f}"
            )

        try:
            # Create GARCH model
            garch_model = arch_model(
                train_data,
                mean=self.mean,
                vol=self.vol,
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=self.rescale,
            )

            # Fit model
            self.model = garch_model.fit(disp="off", show_warning=False)
            self._last_train_data = train_data
            self.is_fitted = True

            logger.debug(
                f"Fitted {self.name} GARCH({self.p},{self.q}) on {len(train_data)} observations"
            )
            logger.debug(f"Model AIC: {self.model.aic:.2f}, BIC: {self.model.bic:.2f}")

        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")
            raise ValueError(f"GARCH fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate volatility forecasts.

        Args:
            horizon: Number of steps ahead

        Returns:
            Series of VOLATILITY predictions (standard deviation, not returns)

        Note:
            - Returns are conditional standard deviations (volatility)
            - To get variance: square the output
            - These are annualized if input was annualized returns
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        try:
            # Forecast volatility
            forecasts = self.model.forecast(horizon=horizon, reindex=False)

            # Extract variance forecasts and convert to standard deviation
            variance_forecast = forecasts.variance.values[-1, :]  # Last row = forecast
            volatility_forecast = np.sqrt(variance_forecast)

            # Generate future dates using base class helper
            last_date = self._last_train_data.index[-1]
            future_dates = self._make_future_index(last_date, horizon)

            predictions = pd.Series(data=volatility_forecast[:horizon], index=future_dates)

            logger.debug(
                f"Generated {horizon}-step GARCH volatility forecast "
                f"(mean vol: {volatility_forecast[:horizon].mean():.4f})"
            )
            return predictions

        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            raise RuntimeError(f"GARCH prediction failed: {e}") from e

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get fitted conditional volatility for training period.

        Returns:
            Series of conditional volatility values for training period

        Use case:
            - Analyze volatility clustering
            - Understand model fit
            - Plot fitted vs actual volatility
        """
        self._validate_fitted()

        cond_vol = self.model.conditional_volatility
        return pd.Series(data=cond_vol, index=self._last_train_data.index)


@register_forecaster("egarch")
class EGARCHForecaster(BaseForecaster):
    """
    EGARCH (Exponential GARCH) volatility forecaster.

    EGARCH handles the "leverage effect" - negative returns tend to
    increase volatility more than positive returns of the same magnitude.

    Advantages over standard GARCH:
    - No parameter constraints (can't be negative)
    - Models asymmetric volatility response
    - Better for equity returns (leverage effect)

    Use cases:
    - Equity volatility forecasting
    - When leverage effect is present
    - More flexible than standard GARCH

    Config:
        p: int - GARCH order (default: 1)
        q: int - ARCH order (default: 1)
        mean: str - Mean model (default: "Constant")
        dist: str - Error distribution (default: "normal")
    """

    family = "volatility"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.p = self.config.get("p", 1)
        self.q = self.config.get("q", 1)
        self.mean = self.config.get("mean", "Constant")
        self.dist = self.config.get("dist", "normal")
        self.model: ARCHModelResult | None = None

    def fit(self, train_data: pd.Series) -> "EGARCHForecaster":
        """Fit EGARCH model on returns data."""
        self._validate_train_data(train_data)

        min_obs = max(30, 10 * (self.p + self.q))
        if len(train_data) < min_obs:
            logger.warning(
                f"EGARCH typically requires >{min_obs} observations. Got {len(train_data)}."
            )

        try:
            garch_model = arch_model(
                train_data,
                mean=self.mean,
                vol="EGARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=True,
            )

            self.model = garch_model.fit(disp="off", show_warning=False)
            self._last_train_data = train_data
            self.is_fitted = True

            logger.debug(f"Fitted EGARCH({self.p},{self.q}) on {len(train_data)} observations")

        except Exception as e:
            logger.error(f"EGARCH fitting failed: {e}")
            raise ValueError(f"EGARCH fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """Generate volatility forecasts using EGARCH."""
        self._validate_fitted()
        self._validate_horizon(horizon)

        try:
            # EGARCH only supports analytic forecasts for horizon=1
            # For horizon>1, use simulation method
            method = "analytic" if horizon == 1 else "simulation"
            forecasts = self.model.forecast(horizon=horizon, method=method, reindex=False)
            variance_forecast = forecasts.variance.values[-1, :]
            volatility_forecast = np.sqrt(variance_forecast)

            # Generate future dates using base class helper
            last_date = self._last_train_data.index[-1]
            future_dates = self._make_future_index(last_date, horizon)

            predictions = pd.Series(data=volatility_forecast[:horizon], index=future_dates)

            logger.debug(
                f"Generated {horizon}-step EGARCH volatility forecast using {method} method"
            )
            return predictions

        except Exception as e:
            logger.error(f"EGARCH prediction failed: {e}")
            raise RuntimeError(f"EGARCH prediction failed: {e}") from e


@register_forecaster("gjr_garch")
class GJRGARCHForecaster(BaseForecaster):
    """
    GJR-GARCH (Threshold GARCH) volatility forecaster.

    Also known as GJR-GARCH or TGARCH.
    Adds a threshold parameter to standard GARCH to capture asymmetric
    volatility response (leverage effect).

    Model adds term: γ * ε²(t-1) * I(ε(t-1) < 0)
    where I() is indicator function for negative shocks.

    Use cases:
    - Model leverage effect (bad news increases vol more)
    - Equity market volatility
    - When asymmetry is important

    Config:
        p: int - GARCH order (default: 1)
        o: int - Asymmetric term order (default: 1)
        q: int - ARCH order (default: 1)
        mean: str - Mean model (default: "Constant")
        dist: str - Error distribution (default: "normal")
    """

    family = "volatility"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.p = self.config.get("p", 1)
        self.o = self.config.get("o", 1)
        self.q = self.config.get("q", 1)
        self.mean = self.config.get("mean", "Constant")
        self.dist = self.config.get("dist", "normal")
        self.model: ARCHModelResult | None = None

    def fit(self, train_data: pd.Series) -> "GJRGARCHForecaster":
        """Fit GJR-GARCH model on returns data."""
        self._validate_train_data(train_data)

        min_obs = max(30, 10 * (self.p + self.o + self.q))
        if len(train_data) < min_obs:
            logger.warning(
                f"GJR-GARCH typically requires >{min_obs} observations. Got {len(train_data)}."
            )

        try:
            garch_model = arch_model(
                train_data,
                mean=self.mean,
                vol="GARCH",
                p=self.p,
                o=self.o,  # Asymmetric term
                q=self.q,
                dist=self.dist,
                rescale=True,
            )

            self.model = garch_model.fit(disp="off", show_warning=False)
            self._last_train_data = train_data
            self.is_fitted = True

            logger.debug(
                f"Fitted GJR-GARCH({self.p},{self.o},{self.q}) on {len(train_data)} observations"
            )

        except Exception as e:
            logger.error(f"GJR-GARCH fitting failed: {e}")
            raise ValueError(f"GJR-GARCH fitting failed: {e}") from e

        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """Generate volatility forecasts using GJR-GARCH."""
        self._validate_fitted()
        self._validate_horizon(horizon)

        try:
            forecasts = self.model.forecast(horizon=horizon, reindex=False)
            variance_forecast = forecasts.variance.values[-1, :]
            volatility_forecast = np.sqrt(variance_forecast)

            # Generate future dates using base class helper
            last_date = self._last_train_data.index[-1]
            future_dates = self._make_future_index(last_date, horizon)

            predictions = pd.Series(data=volatility_forecast[:horizon], index=future_dates)

            logger.debug(f"Generated {horizon}-step GJR-GARCH volatility forecast")
            return predictions

        except Exception as e:
            logger.error(f"GJR-GARCH prediction failed: {e}")
            raise RuntimeError(f"GJR-GARCH prediction failed: {e}") from e


__all__ = [
    "GARCHForecaster",
    "EGARCHForecaster",
    "GJRGARCHForecaster",
]
