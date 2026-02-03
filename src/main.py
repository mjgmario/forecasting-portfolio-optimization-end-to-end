"""
Main Entry Point for Portfolio Forecasting and Optimization
============================================================

This is the main CLI interface for the portfolio forecasting system.
All configuration is loaded from config/default.yaml (or config/local.yaml).

Usage:
    # Single model run with defaults from config
    $ python -m src.main

    # Single model run with specific forecaster/allocator
    $ python -m src.main --forecaster xgboost --allocator robust_markowitz

    # Custom tickers
    $ python -m src.main --tickers AAPL MSFT NVDA

    # Compare multiple forecasters
    $ python -m src.main --compare-forecasters

    # Run comprehensive analysis (all forecasters x all allocators)
    $ python -m src.main --comprehensive --horizon 21

    # List available models
    $ python -m src.main --list-forecasters
    $ python -m src.main --list-allocators

    # Use custom config file
    $ python -m src.main --config config/my_config.yaml

Configuration:
    All settings are loaded from config/default.yaml by default.
    Create config/local.yaml to override settings locally (gitignored).

    Key configuration sections:
    - portfolio.tickers: List of stock tickers
    - portfolio.risk: Risk parameters (min/max allocation, risk aversion)
    - forecasters.default: Default forecaster to use
    - forecasters.skip: Forecasters to skip in comprehensive mode
    - forecasters.configs: Per-forecaster configuration
    - allocators.default: Default allocator to use
    - allocators.configs: Per-allocator configuration
    - backtesting: Walk-forward backtest settings
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.allocators import AllocatorRegistry
from src.config import Config, get_config
from src.constants import MAX_HORIZON, TRADING_DAYS_PER_YEAR
from src.database import get_db
from src.evaluation import BacktestConfig, compare_forecasters
from src.extractor import extract_data
from src.forecasters import ForecasterRegistry
from src.processor import collect_recent_prices, preprocess_data

# =============================================================================
# INPUT VALIDATION
# =============================================================================


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_horizon(horizon: int) -> None:
    """
    Validate forecast horizon is within acceptable range.

    :param horizon: Forecast horizon in days
    :raises ValidationError: If horizon is invalid
    """
    if horizon < 1:
        raise ValidationError(f"Horizon must be at least 1, got {horizon}")
    if horizon > MAX_HORIZON:
        raise ValidationError(
            f"Horizon {horizon} exceeds maximum allowed ({MAX_HORIZON}). "
            "Use a smaller horizon or update MAX_HORIZON in constants.py"
        )


def validate_forecaster(forecaster_name: str) -> None:
    """
    Validate forecaster name is registered.

    :param forecaster_name: Name of the forecaster
    :raises ValidationError: If forecaster is not registered
    """
    if not ForecasterRegistry.is_registered(forecaster_name):
        available = ForecasterRegistry.list_all()
        raise ValidationError(
            f"Unknown forecaster '{forecaster_name}'. "
            f"Available forecasters: {', '.join(sorted(available))}"
        )


def validate_allocator(allocator_name: str) -> None:
    """
    Validate allocator name is registered.

    :param allocator_name: Name of the allocator
    :raises ValidationError: If allocator is not registered
    """
    available = AllocatorRegistry.list_all()
    if allocator_name not in available:
        raise ValidationError(
            f"Unknown allocator '{allocator_name}'. "
            f"Available allocators: {', '.join(sorted(available))}"
        )


def validate_tickers(tickers: list[str]) -> None:
    """
    Validate ticker list is not empty and contains valid symbols.

    :param tickers: List of ticker symbols
    :raises ValidationError: If tickers are invalid
    """
    if not tickers:
        raise ValidationError("At least one ticker is required")

    # Basic validation - tickers should be uppercase alphanumeric
    invalid_tickers = []
    for ticker in tickers:
        if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
            invalid_tickers.append(ticker)

    if invalid_tickers:
        raise ValidationError(
            f"Invalid ticker symbols: {invalid_tickers}. "
            "Tickers should be alphanumeric (e.g., AAPL, MSFT, BRK.B)"
        )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def select_best_forecaster(
    price_series: pd.Series,
    candidates: list[str],
    config: Config,
    metric: str = "directional_accuracy",
    backtest_steps: int = 20,
) -> tuple[str, dict[str, float]]:
    """
    Select the best forecaster from candidates using walk-forward backtesting.

    :param price_series: Time series to test on
    :param candidates: List of forecaster names to evaluate
    :param config: Configuration instance
    :param metric: Metric to use for selection (directional_accuracy, mase, rmse, mae)
    :param backtest_steps: Number of backtest steps
    :return: Tuple of (best_forecaster_name, metrics_dict)

    The best forecaster is selected based on:
    - directional_accuracy: Higher is better
    - mase, rmse, mae: Lower is better
    """
    from src.evaluation import BacktestConfig, walk_forward_backtest
    from src.evaluation.metrics import evaluate_forecast

    logger.info(f"Selecting best forecaster from {len(candidates)} candidates...")
    logger.info(f"Selection metric: {metric}, Backtest steps: {backtest_steps}")

    backtest_config = BacktestConfig(
        initial_train_size=min(len(price_series) - backtest_steps - 10, TRADING_DAYS_PER_YEAR),
        test_size=1,
        step_size=max(1, (len(price_series) - TRADING_DAYS_PER_YEAR) // backtest_steps),
        min_train_size=60,
    )

    results = {}
    for forecaster_name in candidates:
        if config.should_skip_forecaster(forecaster_name):
            logger.debug(f"Skipping {forecaster_name} (in skip list)")
            continue

        if not ForecasterRegistry.is_registered(forecaster_name):
            logger.warning(f"Forecaster {forecaster_name} not registered, skipping")
            continue

        forecaster_config = config.get_forecaster_config(forecaster_name)

        try:
            backtest_results = walk_forward_backtest(
                data=price_series,
                forecaster_name=forecaster_name,
                forecaster_config=forecaster_config,
                config=backtest_config,
            )

            if backtest_results is not None and len(backtest_results.predictions) > 0:
                metrics = backtest_results.metrics or evaluate_forecast(
                    backtest_results.actuals, backtest_results.predictions
                )
                results[forecaster_name] = metrics
                logger.info(
                    f"  {forecaster_name}: DA={metrics.get('directional_accuracy', 0):.3f}, "
                    f"MASE={metrics.get('mase', 0):.3f}"
                )
        except Exception as e:
            logger.warning(f"  {forecaster_name} failed: {e}")
            continue

    if not results:
        logger.warning("No forecaster completed successfully, using default")
        return config.forecasters.default, {}

    # Select best based on metric
    if metric == "directional_accuracy":
        # Higher is better
        best = max(results.items(), key=lambda x: x[1].get(metric, 0))
    else:
        # Lower is better (mase, rmse, mae)
        best = min(results.items(), key=lambda x: x[1].get(metric, float("inf")))

    best_name, best_metrics = best
    logger.info(f"Best forecaster: {best_name} ({metric}={best_metrics.get(metric, 0):.4f})")

    return best_name, results


def _extract_and_preprocess(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """
    Extract historical data and preprocess it for forecasting.

    :param tickers: List of stock tickers
    :param start_date: Start date for data extraction
    :param end_date: End date for data extraction
    :return: Preprocessed portfolio data dictionary
    :raises ValueError: If no data could be extracted
    """
    logger.info("Extracting historical data...")
    all_stock_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not all_stock_data:
        raise ValueError("No data extracted")

    logger.info("Preprocessing data...")
    portfolio_data = preprocess_data(all_stock_data)
    return portfolio_data


def _select_forecaster_for_run(
    portfolio_data: dict[str, Any],
    tickers: list[str],
    forecaster_name: str | None,
    config: Config,
) -> str:
    """
    Determine which forecaster to use based on mode and configuration.

    :param portfolio_data: Preprocessed data dictionary
    :param tickers: List of tickers
    :param forecaster_name: User-specified forecaster (if any)
    :param config: Configuration instance
    :return: Selected forecaster name
    """
    if forecaster_name:
        logger.info(f"Using specified forecaster: {forecaster_name}")
        return forecaster_name

    if config.forecasters.mode == "auto":
        logger.info("Auto-selecting best forecaster...")
        representative_ticker = tickers[0]
        ticker_data = portfolio_data.get(representative_ticker)

        if ticker_data is not None:
            if isinstance(ticker_data, pd.DataFrame):
                price_series = ticker_data["Price"]
            else:
                price_series = ticker_data

            selected, _ = select_best_forecaster(
                price_series=price_series,
                candidates=config.forecasters.candidates,
                config=config,
                metric=config.forecasters.selection_metric,
                backtest_steps=config.forecasters.selection_backtest_steps,
            )
            return selected

        logger.warning(f"Could not auto-select, using default: {config.forecasters.default}")
        return config.forecasters.default

    # Single mode - use default
    logger.info(f"Using default forecaster: {config.forecasters.default}")
    return config.forecasters.default


def _forecast_combined_mode(
    portfolio_data: dict[str, Any],
    tickers: list[str],
    forecaster_name: str,
    forecaster_config: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Generate forecasts using combined training mode (one model for all tickers).

    :param portfolio_data: Preprocessed data dictionary
    :param tickers: List of tickers
    :param forecaster_name: Name of forecaster to use
    :param forecaster_config: Forecaster configuration
    :return: Tuple of (predictions dict, predicted_returns dict)
    :raises ValueError: If combined training fails
    """
    logger.info(f"Training combined model with {forecaster_name}...")

    # Collect all price series
    all_prices = []
    for ticker in tickers:
        if ticker not in portfolio_data:
            continue
        ticker_data = portfolio_data[ticker]
        prices = ticker_data["Price"] if isinstance(ticker_data, pd.DataFrame) else ticker_data
        all_prices.append(prices)

    if not all_prices:
        raise ValueError("No price data available for combined training")

    # Concatenate for combined training
    combined_prices = pd.concat(all_prices, ignore_index=True)
    # Use the earliest date from actual data or today minus the number of periods
    earliest_date = min(
        p.index.min() for p in all_prices if hasattr(p, "index") and len(p.index) > 0
    )
    combined_prices.index = pd.date_range(
        start=earliest_date, periods=len(combined_prices), freq="D"
    )

    # Train combined model (for learning cross-asset patterns)
    forecaster = ForecasterRegistry.create(forecaster_name, config=forecaster_config)
    forecaster.fit(combined_prices)

    # Generate predictions for each ticker
    predictions: dict[str, float] = {}
    predicted_returns: dict[str, float] = {}
    logger.info("  Generating predictions for each ticker...")

    for ticker in tickers:
        if ticker not in portfolio_data:
            continue

        ticker_data = portfolio_data[ticker]
        price_series = (
            ticker_data["Price"] if isinstance(ticker_data, pd.DataFrame) else ticker_data
        )

        try:
            # Refit on ticker-specific data for prediction
            forecaster_ticker = ForecasterRegistry.create(forecaster_name, config=forecaster_config)
            forecaster_ticker.fit(price_series)
            forecast = forecaster_ticker.predict(horizon=1)

            predicted_price = float(forecast.iloc[0])
            current_price = float(price_series.iloc[-1])
            predicted_return = (predicted_price - current_price) / current_price

            predictions[ticker] = predicted_price
            predicted_returns[ticker] = predicted_return

            logger.info(
                f"  {ticker}: ${current_price:.2f} -> ${predicted_price:.2f} "
                f"({predicted_return*100:+.2f}%)"
            )
        except Exception as e:
            logger.error(f"Forecasting failed for {ticker}: {e}")
            continue

    if not predictions:
        raise ValueError("No successful predictions in combined mode")

    return predictions, predicted_returns


def _forecast_individual_mode(
    portfolio_data: dict[str, Any],
    tickers: list[str],
    selected_forecaster: str,
    forecaster_name_override: str | None,
    config: Config,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Generate forecasts using individual training mode (separate model per ticker).

    :param portfolio_data: Preprocessed data dictionary
    :param tickers: List of tickers
    :param selected_forecaster: Default selected forecaster
    :param forecaster_name_override: User-specified forecaster override
    :param config: Configuration instance
    :return: Tuple of (predictions dict, predicted_returns dict)
    """
    logger.info("Forecasting with individual models...")

    predictions: dict[str, float] = {}
    predicted_returns: dict[str, float] = {}

    for ticker in tickers:
        if ticker not in portfolio_data:
            logger.warning(f"Ticker {ticker} not in portfolio data, skipping")
            continue

        ticker_data = portfolio_data[ticker]
        price_series = (
            ticker_data["Price"] if isinstance(ticker_data, pd.DataFrame) else ticker_data
        )

        # Determine forecaster for this ticker
        ticker_forecaster_name = config.forecasters.get_forecaster_for_ticker(ticker)
        if forecaster_name_override:
            ticker_forecaster_name = forecaster_name_override
        elif ticker_forecaster_name != selected_forecaster:
            logger.info(f"  {ticker}: Using per-ticker forecaster: {ticker_forecaster_name}")

        ticker_forecaster_config = config.get_forecaster_config(ticker_forecaster_name)

        try:
            forecaster = ForecasterRegistry.create(
                ticker_forecaster_name, config=ticker_forecaster_config
            )
            forecaster.fit(price_series)
            forecast = forecaster.predict(horizon=1)

            predicted_price = float(forecast.iloc[0])
            current_price = float(price_series.iloc[-1])
            predicted_return = (predicted_price - current_price) / current_price

            predictions[ticker] = predicted_price
            predicted_returns[ticker] = predicted_return

            logger.info(
                f"  {ticker}: ${current_price:.2f} -> ${predicted_price:.2f} "
                f"({predicted_return*100:+.2f}%)"
            )
        except Exception as e:
            logger.error(f"Forecasting failed for {ticker}: {e}")
            continue

    return predictions, predicted_returns


def _optimize_portfolio_weights(
    predicted_returns: dict[str, float],
    portfolio_data: dict[str, Any],
    allocator_name: str,
    allocator_config: dict[str, Any] | None,
) -> dict[str, float]:
    """
    Run portfolio optimization to get optimal weights.

    :param predicted_returns: Dictionary of ticker -> predicted return
    :param portfolio_data: Preprocessed data dictionary
    :param allocator_name: Name of allocator to use
    :param allocator_config: Allocator configuration
    :return: Dictionary of ticker -> weight
    :raises ValueError: If optimization fails
    """
    logger.info(f"Optimizing portfolio with {allocator_name}...")

    predicted_returns_series = pd.Series(predicted_returns)

    # Build historical returns DataFrame
    historical_returns = pd.DataFrame()
    for ticker in predicted_returns.keys():
        ticker_data = portfolio_data[ticker]
        if isinstance(ticker_data, pd.DataFrame):
            historical_returns[ticker] = ticker_data["Price"].pct_change()
        else:
            historical_returns[ticker] = ticker_data.pct_change()
    historical_returns = historical_returns.dropna()

    allocator = AllocatorRegistry.create(allocator_name, config=allocator_config)
    weights = allocator.allocate(
        predicted_returns=predicted_returns_series,
        historical_returns=historical_returns,
    )

    logger.info("\nOptimal Portfolio Weights:")
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ticker}: {weight*100:.2f}%")

    return weights


def _save_forecast_to_db(
    results: dict[str, Any],
    forecaster_name: str,
    allocator_name: str,
    forecaster_config: dict[str, Any] | None,
    allocator_config: dict[str, Any] | None,
) -> None:
    """
    Save forecast and portfolio results to database.

    :param results: Results dictionary
    :param forecaster_name: Name of forecaster used
    :param allocator_name: Name of allocator used
    :param forecaster_config: Forecaster configuration
    :param allocator_config: Allocator configuration
    """
    logger.info("Saving results to database...")
    try:
        db = get_db()
        db.create_tables()
        db.save_forecast_results(results, forecaster_name, forecaster_config)
        db.save_portfolio_results(results, forecaster_name, allocator_name, allocator_config)
        logger.info("Results saved to database")
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")
        logger.warning("Continuing without database save...")


def _build_historical_returns(
    portfolio_data: dict[str, Any],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Build historical returns DataFrame from portfolio data.

    :param portfolio_data: Preprocessed data dictionary
    :param tickers: List of tickers
    :return: DataFrame with historical returns for each ticker
    """
    historical_returns = pd.DataFrame()
    for ticker in tickers:
        ticker_data = portfolio_data.get(ticker)
        if ticker_data is not None:
            if isinstance(ticker_data, pd.DataFrame):
                historical_returns[ticker] = ticker_data["Price"].pct_change()
            else:
                historical_returns[ticker] = ticker_data.pct_change()
    return historical_returns.dropna()


def _run_forecaster_for_all_tickers(
    forecaster_name: str,
    portfolio_data: dict[str, Any],
    tickers: list[str],
    horizon: int,
    config: Config,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Run a single forecaster for all tickers.

    :param forecaster_name: Name of the forecaster
    :param portfolio_data: Preprocessed data dictionary
    :param tickers: List of tickers
    :param horizon: Forecast horizon
    :param config: Configuration instance
    :return: Tuple of (predictions dict, predicted_returns dict)
    """
    forecaster_config = config.get_forecaster_config(forecaster_name)
    predictions: dict[str, float] = {}
    predicted_returns: dict[str, float] = {}

    for ticker in tickers:
        ticker_data = portfolio_data.get(ticker)
        if ticker_data is None:
            continue

        price_series = (
            ticker_data["Price"] if isinstance(ticker_data, pd.DataFrame) else ticker_data
        )

        try:
            forecaster = ForecasterRegistry.create(forecaster_name, config=forecaster_config)
            forecaster.fit(price_series)
            forecast = forecaster.predict(horizon=horizon)

            predicted_price = float(forecast.iloc[-1])
            current_price = float(price_series.iloc[-1])
            predicted_return = (predicted_price - current_price) / current_price

            predictions[ticker] = predicted_price
            predicted_returns[ticker] = predicted_return
        except Exception as e:
            logger.warning(f"    {ticker} failed: {e}")
            continue

    return predictions, predicted_returns


def _run_allocators_for_forecaster(
    forecaster_name: str,
    predicted_returns: dict[str, float],
    historical_returns: pd.DataFrame,
    all_allocators: list[str],
    config: Config,
) -> list[dict[str, Any]]:
    """
    Run all allocators for a single forecaster's predictions.

    :param forecaster_name: Name of the forecaster
    :param predicted_returns: Dictionary of predicted returns
    :param historical_returns: Historical returns DataFrame
    :param all_allocators: List of allocator names
    :param config: Configuration instance
    :return: List of portfolio results
    """
    portfolio_results = []
    predicted_returns_series = pd.Series(predicted_returns)

    for allocator_name in all_allocators:
        allocator_config = config.get_allocator_config(allocator_name)
        try:
            allocator = AllocatorRegistry.create(allocator_name, config=allocator_config)
            weights = allocator.allocate(
                predicted_returns=predicted_returns_series,
                historical_returns=historical_returns[list(predicted_returns.keys())],
            )

            # Determine top allocation
            if hasattr(weights, "idxmax"):
                top_ticker = weights.idxmax()
                top_weight = weights.max()
            else:
                top_ticker = max(weights, key=weights.get)
                top_weight = weights[top_ticker]

            portfolio_results.append(
                {
                    "forecaster": forecaster_name,
                    "allocator": allocator_name,
                    "weights": weights,
                    "predicted_returns": predicted_returns,
                    "top_ticker": top_ticker,
                    "top_weight": float(top_weight) if hasattr(top_weight, "item") else top_weight,
                    "allocator_config": allocator_config,
                }
            )
        except Exception as e:
            logger.warning(f"      {allocator_name} failed: {e}")
            continue

    return portfolio_results


def _print_comprehensive_summary(all_results: dict[str, Any]) -> None:
    """
    Print summary of comprehensive analysis results.

    :param all_results: Dictionary with all results
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info("=" * 80)

    if all_results["summary"]:
        summary_df = pd.DataFrame(all_results["summary"])
        logger.info(f"\nTotal combinations: {len(summary_df)}")
        logger.info(f"Forecasters used: {summary_df['forecaster'].nunique()}")
        logger.info(f"Allocators used: {summary_df['allocator'].nunique()}")

        logger.info("\nTop allocations by forecaster:")
        for forecaster in summary_df["forecaster"].unique():
            row = summary_df[summary_df["forecaster"] == forecaster].iloc[0]
            logger.info(f"  {forecaster}: {row['top_ticker']} ({row['top_weight']*100:.1f}%)")


def prepare_combined_training_data(
    portfolio_data: dict[str, Any],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Prepare combined training data from all tickers.

    Creates a panel dataset with ticker as a feature, allowing the model
    to learn cross-asset patterns.

    :param portfolio_data: Dictionary of ticker -> data
    :param tickers: List of tickers to include
    :return: Combined DataFrame with all tickers' data
    """
    combined_data = []

    for ticker in tickers:
        if ticker not in portfolio_data:
            continue

        ticker_data = portfolio_data[ticker]
        if isinstance(ticker_data, pd.DataFrame):
            df = ticker_data.copy()
        else:
            df = pd.DataFrame({"Price": ticker_data})

        df["ticker"] = ticker
        df["ticker_idx"] = tickers.index(ticker)
        combined_data.append(df)

    if not combined_data:
        return pd.DataFrame()

    combined_df = pd.concat(combined_data, ignore_index=False)
    return combined_df


def run_single_forecast(
    tickers: list[str] | None = None,
    forecaster_name: str | None = None,
    allocator_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    forecaster_config: dict[str, Any] | None = None,
    allocator_config: dict[str, Any] | None = None,
    save_to_db: bool = True,
    training_mode: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """
    Run single forecast + portfolio optimization.

    All parameters default to values from the config file if not provided.

    :param tickers: List of stock tickers (default: from config)
    :param forecaster_name: Name of forecaster (default: from config)
    :param allocator_name: Name of allocator (default: from config)
    :param start_date: Historical data start date (default: from config)
    :param end_date: Historical data end date (default: from config)
    :param forecaster_config: Override forecaster config (default: from config)
    :param allocator_config: Override allocator config (default: from config)
    :param save_to_db: Whether to save results to database
    :param training_mode: "individual" (per ticker) or "combined" (all tickers)
    :param config: Config instance to use
    :return: Results dictionary with predictions and weights

    Example:
        >>> results = run_single_forecast(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     forecaster_name="xgboost",
        ...     allocator_name="robust_markowitz"
        ... )
    """
    if config is None:
        config = get_config()

    # Use config defaults if not provided
    tickers = tickers or config.portfolio.tickers
    allocator_name = allocator_name or config.allocators.default
    start_date = start_date or config.dates.start_date
    end_date = end_date or config.dates.effective_end_date
    training_mode = training_mode or config.forecasters.training_mode
    as_of_date = pd.to_datetime(end_date).date()

    # Log run configuration
    logger.info("=" * 80)
    logger.info(f"Portfolio Optimization Run: {as_of_date}")
    logger.info(f"Tickers: {len(tickers)} assets | Training Mode: {training_mode}")
    logger.info("=" * 80)

    # Step 1-2: Extract and preprocess data
    try:
        portfolio_data = _extract_and_preprocess(tickers, start_date, end_date)
    except ValueError as e:
        logger.error(f"Data extraction failed: {e}")
        return {}

    # Step 3: Select forecaster
    selected_forecaster = _select_forecaster_for_run(
        portfolio_data, tickers, forecaster_name, config
    )

    # Get configs from YAML if not overridden
    if forecaster_config is None:
        forecaster_config = config.get_forecaster_config(selected_forecaster)
    if allocator_config is None:
        allocator_config = config.get_allocator_config(allocator_name)

    logger.info(f"Forecaster: {selected_forecaster} | Allocator: {allocator_name}")

    # Step 4: Generate forecasts
    try:
        if training_mode == "combined":
            predictions, predicted_returns = _forecast_combined_mode(
                portfolio_data, tickers, selected_forecaster, forecaster_config
            )
        else:
            predictions, predicted_returns = _forecast_individual_mode(
                portfolio_data, tickers, selected_forecaster, forecaster_name, config
            )
    except ValueError as e:
        logger.error(f"Forecasting failed: {e}")
        return {}

    if not predictions:
        logger.error("No successful predictions. Exiting.")
        return {}

    # Step 5: Portfolio optimization
    try:
        weights = _optimize_portfolio_weights(
            predicted_returns, portfolio_data, allocator_name, allocator_config
        )
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        return {}

    # Prepare results
    results = {
        "date": as_of_date,
        "forecaster": selected_forecaster,
        "allocator": allocator_name,
        "predictions": predictions,
        "predicted_returns": predicted_returns,
        "weights": weights,
        "actual_prices_last_month": collect_recent_prices(portfolio_data),
    }

    # Step 6: Save to database (optional)
    if save_to_db:
        _save_forecast_to_db(
            results, selected_forecaster, allocator_name, forecaster_config, allocator_config
        )

    logger.info(f"\n{'='*80}")
    logger.info("Portfolio optimization completed successfully!")
    logger.info(f"{'='*80}\n")

    return results


def run_model_comparison(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """
    Compare multiple forecasters using walk-forward backtesting.

    :param tickers: List of stock tickers (default: from config)
    :param start_date: Start date for data (default: from config)
    :param end_date: End date for data (default: from config)
    :param config: Config instance to use
    :return: Comparison DataFrame with metrics
    """
    if config is None:
        config = get_config()

    tickers = tickers or config.portfolio.tickers
    start_date = start_date or config.dates.start_date
    end_date = end_date or config.dates.effective_end_date

    logger.info("=" * 80)
    logger.info("Model Comparison Mode")
    logger.info("=" * 80)

    # Extract data
    logger.info("Extracting historical data...")
    all_stock_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not all_stock_data:
        logger.error("No data extracted.")
        return pd.DataFrame()

    portfolio_data = preprocess_data(all_stock_data)

    # For simplicity, compare on first ticker
    ticker = tickers[0]
    if ticker not in portfolio_data:
        logger.error(f"Ticker {ticker} not found in data")
        return pd.DataFrame()

    ticker_data = portfolio_data[ticker]
    if isinstance(ticker_data, pd.DataFrame):
        price_series = ticker_data["Price"]
    else:
        price_series = ticker_data
    logger.info(f"Running comparison on {ticker} ({len(price_series)} observations)")

    # Get forecasters to compare from config
    forecasters_to_compare = {}
    for name in ForecasterRegistry.list_all():
        if not config.should_skip_forecaster(name):
            forecasters_to_compare[name] = config.get_forecaster_config(name) or None

    # Use backtest config from YAML
    backtest_config = BacktestConfig(
        initial_train_size=config.backtesting.initial_train_size,
        test_size=config.backtesting.test_size,
        step_size=config.backtesting.comparison.get("step_size", 20),
        min_train_size=config.backtesting.min_train_size,
    )

    comparison = compare_forecasters(
        data=price_series,
        forecaster_configs=forecasters_to_compare,
        config=backtest_config,
        verbose=True,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Comparison Results:")
    logger.info("=" * 80)
    print(comparison.to_string())

    # Save to database
    try:
        db = get_db()
        for _, row in comparison.iterrows():
            db.save_backtest_results(
                forecaster=row["forecaster"],
                metrics=row.to_dict(),
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                n_folds=row["n_predictions"],
                config=forecasters_to_compare.get(row["forecaster"]),
            )
        logger.info("Backtest results saved to database")
    except Exception as e:
        logger.error(f"Failed to save backtest results: {e}")

    return comparison


def run_comprehensive_analysis(
    tickers: list[str] | None = None,
    horizon: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    save_to_db: bool = True,
    config: Config | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive analysis with ALL forecasters and ALL allocators.

    Skips forecasters listed in config.forecasters.skip (typically deep learning models).

    :param tickers: List of stock tickers (default: from config)
    :param horizon: Forecast horizon in trading days (default: TRADING_DAYS_PER_YEAR)
    :param start_date: Historical data start date (default: from config)
    :param end_date: Historical data end date (default: from config)
    :param save_to_db: Whether to save results to database
    :param config: Config instance to use
    :return: Dictionary with all results
    """
    if horizon is None:
        horizon = TRADING_DAYS_PER_YEAR
    if config is None:
        config = get_config()

    tickers = tickers or config.portfolio.tickers
    start_date = start_date or config.dates.start_date
    end_date = end_date or config.dates.effective_end_date
    as_of_date = pd.to_datetime(end_date).date()

    # Log run configuration
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ANALYSIS - All Forecasters x All Allocators")
    logger.info("=" * 80)
    logger.info(f"Tickers: {len(tickers)} assets | Horizon: {horizon} days")
    logger.info(f"As of Date: {as_of_date}")

    # Get available forecasters and allocators
    all_forecasters = ForecasterRegistry.list_all()
    all_allocators = AllocatorRegistry.list_all()
    active_forecasters = [f for f in all_forecasters if f not in config.forecasters.skip]

    logger.info(f"Forecasters: {len(active_forecasters)} | Allocators: {len(all_allocators)}")
    logger.info("=" * 80)

    # Step 1-2: Extract and preprocess data
    try:
        portfolio_data = _extract_and_preprocess(tickers, start_date, end_date)
    except ValueError as e:
        logger.error(f"Data extraction failed: {e}")
        return {}

    historical_returns = _build_historical_returns(portfolio_data, tickers)

    # Initialize results
    all_results: dict[str, Any] = {
        "forecaster_results": {},
        "portfolio_results": [],
        "summary": [],
    }

    # Step 3: Run all forecasters
    logger.info(f"\nRunning {len(active_forecasters)} forecasters...")

    for forecaster_name in active_forecasters:
        logger.info(f"\n  Forecaster: {forecaster_name}")

        training_mode = config.forecasters.training_mode
        if training_mode == "combined":
            forecaster_config = config.get_forecaster_config(forecaster_name)
            predictions, predicted_returns = _forecast_combined_mode(
                portfolio_data, tickers, forecaster_name, forecaster_config
            )
        else:
            predictions, predicted_returns = _run_forecaster_for_all_tickers(
                forecaster_name, portfolio_data, tickers, horizon, config
            )

        if not predictions:
            logger.warning(f"  No predictions for {forecaster_name}, skipping allocators")
            continue

        all_results["forecaster_results"][forecaster_name] = {
            "predictions": predictions,
            "predicted_returns": predicted_returns,
        }

        avg_return = sum(predicted_returns.values()) / len(predicted_returns)
        logger.info(f"    Avg predicted return: {avg_return*100:.2f}%")

        # Save forecast results to database
        if save_to_db:
            _save_comprehensive_forecast(
                as_of_date, predictions, predicted_returns, forecaster_name, config
            )

        # Step 4: Run all allocators for this forecaster
        logger.info(f"    Running {len(all_allocators)} allocators...")
        portfolio_results = _run_allocators_for_forecaster(
            forecaster_name, predicted_returns, historical_returns, all_allocators, config
        )

        # Process results and save to database
        for result in portfolio_results:
            all_results["portfolio_results"].append(
                {
                    "forecaster": result["forecaster"],
                    "allocator": result["allocator"],
                    "weights": result["weights"],
                    "predicted_returns": result["predicted_returns"],
                }
            )
            all_results["summary"].append(
                {
                    "forecaster": result["forecaster"],
                    "allocator": result["allocator"],
                    "top_ticker": result["top_ticker"],
                    "top_weight": result["top_weight"],
                    "avg_predicted_return": avg_return,
                }
            )

            if save_to_db:
                _save_comprehensive_portfolio(as_of_date, result, forecaster_name, config)

    _print_comprehensive_summary(all_results)
    return all_results


def _save_comprehensive_forecast(
    as_of_date: Any,
    predictions: dict[str, float],
    predicted_returns: dict[str, float],
    forecaster_name: str,
    config: Config,
) -> None:
    """Save forecast results during comprehensive analysis."""
    try:
        db = get_db()
        db.create_tables()
        forecast_result = {
            "date": as_of_date,
            "predictions": predictions,
            "predicted_returns": predicted_returns,
        }
        forecaster_config = config.get_forecaster_config(forecaster_name)
        db.save_forecast_results(forecast_result, forecaster_name, forecaster_config)
    except Exception as e:
        logger.warning(f"    Failed to save forecasts: {e}")


def _save_comprehensive_portfolio(
    as_of_date: Any,
    result: dict[str, Any],
    forecaster_name: str,
    config: Config,
) -> None:
    """Save portfolio results during comprehensive analysis."""
    try:
        db = get_db()
        portfolio_result = {"date": as_of_date, "weights": result["weights"]}
        db.save_portfolio_results(
            portfolio_result, forecaster_name, result["allocator"], result["allocator_config"]
        )
    except Exception as e:
        logger.warning(f"      Failed to save portfolio: {e}")


def main():
    """
    Main CLI entry point.

    All settings are loaded from config/default.yaml (or config/local.yaml).
    CLI arguments are optional overrides.
    """
    parser = argparse.ArgumentParser(
        description="Portfolio Forecasting and Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All settings are loaded from config/default.yaml.
Create config/local.yaml for personal overrides (gitignored).

CLI arguments override config values when provided.

Examples:
  # Run with all defaults from config
  python -m src.main

  # Override forecaster (config value takes precedence otherwise)
  python -m src.main --forecaster xgboost

  # List available models
  python -m src.main --list-forecasters
  python -m src.main --list-allocators
        """,
    )

    # Override arguments (all optional, config values used if not provided)
    parser.add_argument(
        "--forecaster",
        type=str,
        default=None,
        help="Override forecaster (default: from config)",
    )

    parser.add_argument(
        "--allocator",
        type=str,
        default=None,
        help="Override allocator (default: from config)",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Override tickers (default: from config)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config directory",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile to load (e.g., full_comparison, production, debug)",
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["individual", "combined"],
        default=None,
        help="Override training mode (default: from config)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "compare", "comprehensive"],
        default=None,
        help="Override execution mode (default: from config)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override forecast horizon (default: from config)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Override save_to_db to False",
    )

    # Utility arguments
    parser.add_argument(
        "--list-forecasters",
        action="store_true",
        help="List available forecasters and exit",
    )

    parser.add_argument(
        "--list-allocators",
        action="store_true",
        help="List available allocators and exit",
    )

    args = parser.parse_args()

    # Load config
    profile = args.profile or "default"
    if args.config:
        config = Config(config_dir=Path(args.config), profile=profile)
    else:
        config = get_config(profile=profile)

    # List forecasters
    if args.list_forecasters:
        print("\nAvailable Forecasters:")
        print("=" * 60)
        for family in ForecasterRegistry.get_families():
            models = ForecasterRegistry.list_by_family(family)
            print(f"\n{family.upper()}:")
            for model in models:
                skip_marker = " [SKIPPED]" if config.should_skip_forecaster(model) else ""
                default_marker = " [DEFAULT]" if model == config.forecasters.default else ""
                print(f"  - {model}{default_marker}{skip_marker}")
        print("\n")
        return

    # List allocators
    if args.list_allocators:
        print("\nAvailable Allocators:")
        print("=" * 60)
        for allocator in AllocatorRegistry.list_all():
            info = AllocatorRegistry.get_info(allocator)
            default_marker = " [DEFAULT]" if allocator == config.allocators.default else ""
            print(f"  - {allocator:<25}{default_marker} ({info['class_name']})")
        print("\n")
        return

    # Determine execution mode from config (with CLI override)
    execution_mode = args.mode or config.execution.mode
    save_to_db = config.execution.save_to_db and not args.no_save
    horizon = args.horizon or config.execution.horizon

    # Validate inputs
    try:
        if args.horizon is not None:
            validate_horizon(args.horizon)
        if args.forecaster is not None:
            validate_forecaster(args.forecaster)
        if args.allocator is not None:
            validate_allocator(args.allocator)
        if args.tickers is not None:
            validate_tickers(args.tickers)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Configuration Summary:")
    logger.info(f"  Execution mode: {execution_mode}")
    logger.info(f"  Forecaster: {args.forecaster or config.forecasters.default}")
    logger.info(f"  Allocator: {args.allocator or config.allocators.default}")
    logger.info(f"  Training mode: {args.training_mode or config.forecasters.training_mode}")
    logger.info(f"  Tickers: {len(args.tickers or config.portfolio.tickers)} assets")
    logger.info(f"  Save to DB: {save_to_db}")
    logger.info("=" * 80)

    # Run based on execution mode
    if execution_mode == "compare":
        run_model_comparison(tickers=args.tickers, config=config)
        return

    if execution_mode == "comprehensive":
        run_comprehensive_analysis(
            tickers=args.tickers,
            horizon=horizon,
            save_to_db=save_to_db,
            config=config,
        )
        return

    # Default: single mode
    try:
        results = run_single_forecast(
            tickers=args.tickers,
            forecaster_name=args.forecaster,
            allocator_name=args.allocator,
            save_to_db=save_to_db,
            training_mode=args.training_mode,
            config=config,
        )

        if not results:
            logger.error("Optimization failed")
            sys.exit(1)

        print("\n" + "=" * 80)
        print("SUCCESS: Portfolio optimization completed")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
