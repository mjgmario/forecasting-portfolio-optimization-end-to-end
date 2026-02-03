"""
Evaluation Module

Tools for evaluating forecasting and portfolio performance:
- Metrics: MAE, RMSE, MASE, Directional Accuracy, etc.
- Backtesting: Walk-forward backtesting framework
- Model Comparison: Compare multiple forecasters

Usage:
    >>> from src.evaluation import walk_forward_backtest, evaluate_forecast
    >>> result = walk_forward_backtest(data, "xgboost")
    >>> print(result.summary())
"""

from .backtesting import (
    BacktestConfig,
    BacktestResult,
    backtest_portfolio_strategy,
    compare_forecasters,
    walk_forward_backtest,
)
from .metrics import (
    directional_accuracy,
    evaluate_forecast,
    hit_rate,
    information_ratio,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    tracking_error,
)

__all__ = [
    # Metrics
    "mae",
    "rmse",
    "mape",
    "mase",
    "directional_accuracy",
    "hit_rate",
    "pinball_loss",
    "tracking_error",
    "information_ratio",
    "evaluate_forecast",
    # Backtesting
    "BacktestConfig",
    "BacktestResult",
    "walk_forward_backtest",
    "compare_forecasters",
    "backtest_portfolio_strategy",
]
