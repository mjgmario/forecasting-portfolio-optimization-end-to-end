"""
Forecasters Module
==================

This module provides a plugin-based architecture for forecasting models.
All forecasters inherit from BaseForecaster and are managed through the ForecasterRegistry.

Available forecasters:
- Baseline: naive, seasonal_naive, drift
- Statistical: ets, theta, arima, prophet
- ML: ridge, lasso, elasticnet, random_forest, hist_gradient_boosting, xgboost, lightgbm, catboost
- Volatility: garch, egarch, gjr_garch
- Deep Learning: lstm, transformer

Usage:
    >>> from src.forecasters import ForecasterRegistry
    >>> forecaster = ForecasterRegistry.create("xgboost", config={"n_estimators": 200})
    >>> forecaster.fit(train_data)
    >>> prediction = forecaster.predict_point()
"""

# Import all forecaster modules to trigger registration (flat structure)
from . import (
    baselines,  # noqa: F401 - NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster
    deep_learning,  # noqa: F401 - LSTMForecaster, TransformerForecaster
    ml_regressors,  # noqa: F401 - Ridge, Lasso, XGBoost, etc.
    prophet,  # noqa: F401 - ProphetForecaster
    statistical,  # noqa: F401 - ETSForecaster, ThetaForecaster, ARIMAForecaster
    volatility,  # noqa: F401 - GARCHForecaster, EGARCHForecaster
)
from .base import BaseForecaster
from .registry import ForecasterRegistry, get_forecaster, register_forecaster

__all__ = [
    "BaseForecaster",
    "ForecasterRegistry",
    "get_forecaster",
    "register_forecaster",
]
