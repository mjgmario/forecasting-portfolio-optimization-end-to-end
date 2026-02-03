"""
Walk-Forward Backtesting Engine

Implements walk-forward (rolling window) backtesting for time series forecasting.

Why walk-forward:
1. Mimics real-world forecasting (train on past, predict future)
2. Avoids look-ahead bias
3. Tests model performance over time
4. More realistic than train/test split

Process:
1. Start with initial training window
2. Train model, predict next period
3. Roll window forward, retrain
4. Repeat until end of data
5. Evaluate all predictions

"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..forecasters import ForecasterRegistry
from .metrics import directional_accuracy_1step, evaluate_forecast

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtest."""

    initial_train_size: int = 252  # ~1 year of trading days
    test_size: int = 1  # Forecast 1 step ahead
    step_size: int = 1  # Roll forward 1 day at a time
    min_train_size: int = 60  # Minimum training data required
    refit_every: int = 1  # Re-train every N steps (1=always, 5=weekly, 21=monthly)
    gap_size: int = 0  # Gap between train end and test start (anti-leakage)

    def validate(self):
        """Validate configuration."""
        if self.initial_train_size < self.min_train_size:
            raise ValueError(
                f"initial_train_size ({self.initial_train_size}) must be >= "
                f"min_train_size ({self.min_train_size})"
            )
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.gap_size < 0:
            raise ValueError("gap_size must be >= 0")


@dataclass
class FoldResult:
    """Result from a single backtest fold."""

    fold_idx: int
    train_end_date: pd.Timestamp | None = None
    y_true: float | np.ndarray = None
    y_pred: float | np.ndarray = None
    y_prev: float = None  # Last value from train (for directional accuracy 1-step)
    fold_metrics: dict[str, float] = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    failed: bool = False
    error_message: str | None = None


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""

    forecaster_name: str
    predictions: pd.Series  # All predictions
    actuals: pd.Series  # Actual values
    train_sizes: list[int] = None  # Training size for each prediction
    metrics: dict[str, float] = None  # Evaluation metrics
    errors: pd.Series = None  # Prediction errors
    fold_times: list[float] = None  # Training time for each fold
    # New: detailed fold results
    fold_results: list[FoldResult] = None
    failed_folds: list[int] = None
    failure_rate: float = None
    y_prev_values: list[float] = None  # For directional_accuracy_1step
    # Backward compatibility - individual metrics
    mae: float = None
    rmse: float = None
    mase: float = None
    directional_accuracy: float = None
    n_folds: int = None

    def __post_init__(self):
        """Initialize metrics dict from individual parameters if provided."""
        # If metrics dict not provided but individual metrics are, construct it
        if self.metrics is None and any(
            [self.mae, self.rmse, self.mase, self.directional_accuracy]
        ):
            self.metrics = {}
            if self.mae is not None:
                self.metrics["mae"] = self.mae
            if self.rmse is not None:
                self.metrics["rmse"] = self.rmse
            if self.mase is not None:
                self.metrics["mase"] = self.mase
            if self.directional_accuracy is not None:
                self.metrics["directional_accuracy"] = self.directional_accuracy
        elif self.metrics is None:
            self.metrics = {}

        # Set default values for optional fields
        if self.train_sizes is None:
            self.train_sizes = []
        if self.errors is None:
            self.errors = (
                self.predictions - self.actuals
                if self.predictions is not None and self.actuals is not None
                else pd.Series()
            )
        if self.fold_times is None:
            self.fold_times = []
        if self.fold_results is None:
            self.fold_results = []
        if self.failed_folds is None:
            self.failed_folds = []
        if self.y_prev_values is None:
            self.y_prev_values = []

        # Set n_folds if not provided
        if self.n_folds is None:
            self.n_folds = len(self.train_sizes) if self.train_sizes else 0

        # Calculate failure rate
        if self.failure_rate is None and self.fold_results:
            total = len(self.fold_results)
            failed = sum(1 for f in self.fold_results if f.failed)
            self.failure_rate = failed / total if total > 0 else 0.0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Backtest Results: {self.forecaster_name}",
            "=" * 60,
            f"Number of predictions: {len(self.predictions)}",
            f"Average train size: {np.mean(self.train_sizes):.0f}",
            f"Total folds: {self.n_folds}",
        ]

        # Add failure rate if available
        if self.failure_rate is not None and self.failure_rate > 0:
            lines.append(f"Failed folds: {len(self.failed_folds)} ({self.failure_rate*100:.1f}%)")

        lines.extend(["", "Metrics:"])

        for metric_name, value in self.metrics.items():
            # Uppercase metric name for display
            display_name = (
                metric_name.upper()
                if len(metric_name) <= 4
                else metric_name.replace("_", " ").title()
            )

            if "accuracy" in metric_name.lower() or "rate" in metric_name.lower():
                # Display fractions as percentages
                lines.append(f"  {display_name}: {value*100:.2f}%")
            elif "ratio" in metric_name.lower():
                lines.append(f"  {display_name}: {value:.3f}")
            else:
                lines.append(f"  {display_name}: {value:.6f}")

        return "\n".join(lines)

    def get_directional_accuracy(self) -> float:
        """Get directional accuracy if available."""
        return self.metrics.get("directional_accuracy", np.nan)


def walk_forward_backtest(
    data: pd.Series,
    forecaster_name: str,
    forecaster_config: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> BacktestResult:
    """
    Perform walk-forward backtesting for a single forecaster.

    :param data: Time series data (prices or returns) with DatetimeIndex
    :type data: pd.Series
    :param forecaster_name: Name of forecaster from registry
    :type forecaster_name: str
    :param forecaster_config: Configuration for the forecaster
    :type forecaster_config: dict, optional
    :param config: Backtest configuration
    :type config: BacktestConfig, optional
    :param verbose: Show progress bar
    :type verbose: bool
    :return: BacktestResult object with predictions and metrics
    :rtype: BacktestResult

    Example::

        >>> prices = pd.Series(..., index=pd.DatetimeIndex(...))
        >>> config = BacktestConfig(initial_train_size=252, test_size=1)
        >>> result = walk_forward_backtest(
        ...     prices,
        ...     "xgboost",
        ...     forecaster_config={"lags": 10, "n_estimators": 100},
        ...     config=config
        ... )
        >>> print(result.summary())
    """
    if config is None:
        config = BacktestConfig()

    config.validate()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")

    if len(data) < config.initial_train_size + config.test_size:
        raise ValueError(
            f"Data length ({len(data)}) insufficient for backtest. "
            f"Need at least {config.initial_train_size + config.test_size}"
        )

    logger.info(f"Starting walk-forward backtest: {forecaster_name}")
    logger.info(f"  Data length: {len(data)}")
    logger.info(f"  Initial train size: {config.initial_train_size}")
    logger.info(f"  Test size: {config.test_size}")
    logger.info(f"  Step size: {config.step_size}")
    logger.info(f"  Refit every: {config.refit_every} steps")
    logger.info(f"  Gap size: {config.gap_size}")

    # Initialize storage
    predictions_list = []
    actuals_list = []
    dates_list = []
    train_sizes_list = []
    fold_times_list = []
    fold_results_list = []
    failed_folds_list = []
    y_prev_list = []  # For directional_accuracy_1step

    # Calculate number of folds (accounting for gap)
    n_observations = len(data)
    n_folds = (
        n_observations - config.initial_train_size - config.gap_size - config.test_size
    ) // config.step_size + 1

    # Progress bar
    iterator = range(n_folds)
    if verbose:
        iterator = tqdm(iterator, desc=f"Backtesting {forecaster_name}")

    import time

    forecaster = None  # Will be created/refitted as needed

    for i in iterator:
        # Define train/test split with gap
        train_end = config.initial_train_size + i * config.step_size
        test_start = train_end + config.gap_size  # Apply gap
        test_end = test_start + config.test_size

        if test_end > n_observations:
            break

        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]
        y_prev = train_data.iloc[-1]  # Last value from training data

        # Create and fit forecaster (only on first fold or when refit is needed)
        try:
            start_time = time.time()

            # Only refit if: first fold OR refit interval reached
            if forecaster is None or i % config.refit_every == 0:
                forecaster = ForecasterRegistry.create(forecaster_name, config=forecaster_config)
                forecaster.fit(train_data)

                # Warn if forecaster horizon doesn't match test_size (especially for DL models)
                if hasattr(forecaster, "h") and forecaster.h != config.test_size:
                    logger.warning(
                        f"Forecaster horizon (h={forecaster.h}) != backtest test_size ({config.test_size}). "
                        f"Consider setting test_size={forecaster.h} for optimal results with DL models."
                    )

            fit_time = time.time() - start_time
            predict_start = time.time()

            # Predict
            forecast = forecaster.predict(horizon=config.test_size)

            # Force alignment with actual test data index
            # This handles cases where forecaster generates incorrect dates
            # (e.g., weekends for business day data)
            forecast = pd.Series(forecast.values[: len(test_data)], index=test_data.index)

            predict_time = time.time() - predict_start
            fold_time = time.time() - start_time

            # Store results
            predictions_list.extend(forecast.values)
            actuals_list.extend(test_data.values)
            dates_list.extend(test_data.index)
            train_sizes_list.append(len(train_data))
            fold_times_list.append(fold_time)
            y_prev_list.extend([y_prev] * len(test_data))

            # Create FoldResult
            fold_results_list.append(
                FoldResult(
                    fold_idx=i,
                    train_end_date=train_data.index[-1],
                    y_true=test_data.values[0] if len(test_data) == 1 else test_data.values,
                    y_pred=forecast.values[0] if len(forecast) == 1 else forecast.values,
                    y_prev=y_prev,
                    fit_time=fit_time,
                    predict_time=predict_time,
                    failed=False,
                )
            )

        except Exception as e:
            logger.warning(f"Fold {i} failed: {e}")
            # Don't skip - record as failed fold for unbiased metrics
            failed_folds_list.append(i)
            fold_results_list.append(
                FoldResult(
                    fold_idx=i,
                    train_end_date=train_data.index[-1] if len(train_data) > 0 else None,
                    y_true=test_data.values[0] if len(test_data) == 1 else np.nan,
                    y_pred=np.nan,
                    y_prev=y_prev,
                    failed=True,
                    error_message=str(e),
                )
            )
            # Still record actuals for failure tracking (predictions as NaN)
            # Handle both test_size=1 and test_size>1 cases
            if config.test_size == 1:
                predictions_list.append(np.nan)
                actuals_list.append(test_data.values[0])
                dates_list.append(test_data.index[0])
                y_prev_list.append(y_prev)
                train_sizes_list.append(len(train_data))
                fold_times_list.append(0.0)
            else:
                # For multi-step forecasting, record NaN predictions with actual values
                predictions_list.extend([np.nan] * len(test_data))
                actuals_list.extend(test_data.values)
                dates_list.extend(test_data.index)
                y_prev_list.extend([y_prev] * len(test_data))
                train_sizes_list.append(len(train_data))
                fold_times_list.append(0.0)

    if len(predictions_list) == 0 or all(np.isnan(p) for p in predictions_list):
        raise RuntimeError("All backtest folds failed. Cannot compute results.")

    # Create series
    predictions = pd.Series(predictions_list, index=dates_list)
    actuals = pd.Series(actuals_list, index=dates_list)
    y_prev_series = pd.Series(y_prev_list, index=dates_list)

    # Filter out NaN predictions for metric calculation
    valid_mask = ~predictions.isna()
    valid_predictions = predictions[valid_mask]
    valid_actuals = actuals[valid_mask]
    valid_y_prev = y_prev_series[valid_mask]

    # Compute metrics (ignoring failed folds)
    train_data_full = data.iloc[: config.initial_train_size]
    metrics = evaluate_forecast(valid_actuals, valid_predictions, y_train=train_data_full)

    # Add directional_accuracy_1step for walk-forward backtest
    if len(valid_y_prev) > 0:
        try:
            da_1step = directional_accuracy_1step(valid_y_prev, valid_actuals, valid_predictions)
            metrics["directional_accuracy_1step"] = da_1step
        except Exception as e:
            logger.warning(f"directional_accuracy_1step calculation failed: {e}")

    # Compute errors
    errors = actuals - predictions

    # Calculate failure rate
    total_folds = len(fold_results_list)
    failure_rate = len(failed_folds_list) / total_folds if total_folds > 0 else 0.0

    logger.info(f"Backtest complete: {len(valid_predictions)} predictions")
    if failed_folds_list:
        logger.info(
            f"  Failed folds: {len(failed_folds_list)}/{total_folds} ({failure_rate*100:.1f}%)"
        )
    logger.info(f"  MAE: {metrics.get('mae', np.nan):.6f}")
    logger.info(f"  MASE: {metrics.get('mase', np.nan):.3f}")
    logger.info(
        f"  Directional Accuracy 1-step: {metrics.get('directional_accuracy_1step', np.nan)*100:.2f}%"
    )

    return BacktestResult(
        forecaster_name=forecaster_name,
        predictions=predictions,
        actuals=actuals,
        train_sizes=train_sizes_list,
        metrics=metrics,
        errors=errors,
        fold_times=fold_times_list,
        fold_results=fold_results_list,
        failed_folds=failed_folds_list,
        failure_rate=failure_rate,
        y_prev_values=y_prev_list,
        n_folds=total_folds,
    )


def compare_forecasters(
    data: pd.Series,
    forecaster_configs: dict[str, dict[str, Any] | None],
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare multiple forecasters using walk-forward backtesting.

    :param data: Time series data
    :type data: pd.Series
    :param forecaster_configs: Dict mapping forecaster names to their configs
    :type forecaster_configs: dict
    :param config: Backtest configuration
    :type config: BacktestConfig, optional
    :param verbose: Show progress
    :type verbose: bool
    :return: DataFrame comparing forecasters with all metrics
    :rtype: pd.DataFrame

    Example::

        >>> forecasters = {
        ...     "naive": None,
        ...     "arima": {"seasonal": False},
        ...     "xgboost": {"n_estimators": 100, "lags": 10},
        ...     "ensemble_avg": {"forecasters": ["naive", "theta", "arima"]}
        ... }
        >>> comparison = compare_forecasters(prices, forecasters)
        >>> print(comparison.sort_values("mase"))
    """
    if config is None:
        config = BacktestConfig()

    results = []

    for forecaster_name, forecaster_config in forecaster_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting: {forecaster_name}")
        logger.info(f"{'='*60}")

        try:
            result = walk_forward_backtest(
                data=data,
                forecaster_name=forecaster_name,
                forecaster_config=forecaster_config,
                config=config,
                verbose=verbose,
            )

            # Extract metrics
            row = {
                "forecaster": forecaster_name,
                "n_predictions": len(result.predictions),
                **result.metrics,
                "avg_train_time": np.mean(result.fold_times),
            }

            results.append(row)

        except Exception as e:
            logger.error(f"Backtesting {forecaster_name} failed: {e}")
            continue

    if not results:
        raise RuntimeError("All forecasters failed during backtesting")

    # Create comparison dataframe
    df = pd.DataFrame(results)

    # Sort by MASE (best first)
    if "mase" in df.columns:
        df = df.sort_values("mase")

    return df


def backtest_portfolio_strategy(
    prices: pd.DataFrame,
    forecaster_name: str,
    allocator_name: str,
    forecaster_config: dict[str, Any] | None = None,
    allocator_config: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Backtest a complete portfolio strategy (forecaster + allocator).

    For each time period:

    1. Train forecasters on all assets
    2. Generate return forecasts
    3. Use allocator to get portfolio weights
    4. Calculate portfolio return
    5. Roll forward and repeat

    :param prices: DataFrame with prices (dates as index, tickers as columns)
    :type prices: pd.DataFrame
    :param forecaster_name: Name of forecaster
    :type forecaster_name: str
    :param allocator_name: Name of allocator
    :type allocator_name: str
    :param forecaster_config: Forecaster configuration
    :type forecaster_config: dict, optional
    :param allocator_config: Allocator configuration
    :type allocator_config: dict, optional
    :param config: Backtest configuration
    :type config: BacktestConfig, optional
    :param verbose: Show progress
    :type verbose: bool
    :return: Dict with portfolio returns, weights history, and metrics
    :rtype: dict

    Example::
        >>> prices_df = pd.DataFrame(...)  # Prices for multiple assets
        >>> results = backtest_portfolio_strategy(
        ...     prices_df,
        ...     forecaster_name="xgboost",
        ...     allocator_name="robust_markowitz"
        ... )
        >>> portfolio_returns = results["portfolio_returns"]
        >>> sharpe_ratio = results["sharpe_ratio"]
    """
    if config is None:
        config = BacktestConfig()

    logger.info("Starting portfolio backtest...")
    logger.info(f"  Forecaster: {forecaster_name}")
    logger.info(f"  Allocator: {allocator_name}")
    logger.info(f"  Assets: {list(prices.columns)}")

    # This is a simplified version - full implementation would be more complex
    # For now, return placeholder
    logger.warning("Full portfolio backtesting not yet implemented - this is a placeholder")

    return {
        "portfolio_returns": pd.Series([]),
        "weights_history": pd.DataFrame([]),
        "sharpe_ratio": np.nan,
        "total_return": np.nan,
        "max_drawdown": np.nan,
    }


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FoldResult",
    "walk_forward_backtest",
    "compare_forecasters",
    "backtest_portfolio_strategy",
]
