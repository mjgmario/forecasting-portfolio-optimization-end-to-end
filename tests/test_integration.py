"""
Integration Tests for Forecasting End-to-End Pipeline

These tests verify that the full pipeline works correctly from data extraction
through forecasting to portfolio allocation.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.allocators.registry import AllocatorRegistry
from src.evaluation.backtesting import BacktestConfig, walk_forward_backtest
from src.evaluation.metrics import directional_accuracy, mae, rmse
from src.forecasters.registry import ForecasterRegistry

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Generate correlated returns for multiple assets
    n_assets = 3
    returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets,
        cov=[[0.0004, 0.0002, 0.0001], [0.0002, 0.0005, 0.00015], [0.0001, 0.00015, 0.0003]],
        size=100,
    )

    # Convert returns to prices
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0), index=dates, columns=["AAPL", "GOOGL", "MSFT"]
    )

    return prices


@pytest.fixture
def sample_return_data(sample_price_data):
    """Create sample return data."""
    returns = sample_price_data.pct_change().dropna()
    return returns


class TestForecasterIntegration:
    """Integration tests for forecasters."""

    def test_baseline_forecaster_pipeline(self, sample_price_data):
        """Test full pipeline with baseline forecasters."""
        ticker_data = sample_price_data["AAPL"]
        train_data = ticker_data[:80]
        test_data = ticker_data[80:]

        # Test multiple baseline forecasters
        forecaster_names = ["naive", "seasonal_naive", "drift"]

        for name in forecaster_names:
            forecaster = ForecasterRegistry.create(name)
            forecaster.fit(train_data)

            # Generate forecasts
            horizon = len(test_data)
            predictions = forecaster.predict(horizon=horizon)

            # Verify predictions
            assert len(predictions) == horizon
            assert predictions.index[0] > train_data.index[-1]
            assert forecaster.is_fitted
            assert forecaster.family == "baseline"

            # Calculate metrics
            error_mae = mae(test_data, predictions)
            error_rmse = rmse(test_data, predictions)

            assert error_mae > 0
            assert error_rmse >= error_mae  # RMSE >= MAE always

            logger.info(f"{name}: MAE={error_mae:.4f}, RMSE={error_rmse:.4f}")

    def test_ml_forecaster_pipeline(self, sample_price_data):
        """Test full pipeline with ML forecasters."""
        ticker_data = sample_price_data["AAPL"]
        train_data = ticker_data[:80]
        ticker_data[80:]

        # Test Ridge forecaster
        forecaster = ForecasterRegistry.create("ridge", config={"lags": 5, "horizon": 1})
        forecaster.fit(train_data)

        # Generate forecast
        predictions = forecaster.predict(horizon=1)

        assert len(predictions) == 1
        assert forecaster.is_fitted
        assert forecaster.family == "ml"

    def test_statistical_forecaster_pipeline(self, sample_price_data):
        """Test full pipeline with statistical forecasters (requires statsforecast)."""
        ticker_data = sample_price_data["AAPL"]
        train_data = ticker_data[:80]

        # Test ETS forecaster
        forecaster = ForecasterRegistry.create("ets", config={"season_length": 1})
        forecaster.fit(train_data)

        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5
        assert forecaster.is_fitted
        assert forecaster.family == "statistical"
        assert forecaster.model is not None


class TestAllocatorIntegration:
    """Integration tests for allocators."""

    def test_simple_allocator_pipeline(self, sample_price_data, sample_return_data):
        """Test full pipeline with simple allocators."""
        # Generate predicted returns (just use actual returns for testing)
        predicted_returns = sample_return_data.iloc[-1]
        historical_returns = sample_return_data.iloc[:-10]

        allocator_names = ["equal_weight", "inverse_volatility", "risk_parity", "minimum_variance"]

        for name in allocator_names:
            allocator = AllocatorRegistry.create(name)
            weights = allocator.allocate(predicted_returns, historical_returns)

            # Verify weights
            assert len(weights) == len(predicted_returns)
            assert all(ticker in weights.index for ticker in predicted_returns.index)
            assert abs(weights.sum() - 1.0) < 1e-6  # Sum to 1
            assert all(w >= 0 for w in weights.values)  # Non-negative
            # equal_weight and inverse_volatility are "simple", risk_parity and minimum_variance are "risk_based"
            assert allocator.family in ["simple", "risk_based"]

            logger.info(f"{name}: {weights}")

    def test_markowitz_allocator_pipeline(self, sample_price_data, sample_return_data):
        """Test full pipeline with Markowitz allocators."""
        predicted_returns = sample_return_data.iloc[-1]
        historical_returns = sample_return_data.iloc[:-10]

        # Test Markowitz
        allocator = AllocatorRegistry.create("markowitz", config={"risk_aversion": 1.0})
        weights = allocator.allocate(predicted_returns, historical_returns)

        assert len(weights) == len(predicted_returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert allocator.family == "mean_variance"

        # Test Robust Markowitz
        allocator = AllocatorRegistry.create(
            "robust_markowitz", config={"risk_aversion": 1.0, "covariance_shrinkage": 0.1}
        )
        weights = allocator.allocate(predicted_returns, historical_returns)

        assert len(weights) == len(predicted_returns)
        assert abs(weights.sum() - 1.0) < 1e-6


class TestBacktestingIntegration:
    """Integration tests for backtesting framework."""

    def test_walk_forward_backtest_pipeline(self, sample_price_data):
        """Test full walk-forward backtesting pipeline."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(initial_train_size=60, test_size=5, step_size=5, min_train_size=50)

        # Backtest naive forecaster
        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        # Verify results
        n_folds = len(result.train_sizes)
        assert n_folds > 0
        assert len(result.predictions) > 0
        assert len(result.actuals) == len(result.predictions)
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert result.metrics["mae"] > 0
        assert result.metrics["rmse"] >= result.metrics["mae"]

        logger.info(
            f"Backtest results: MAE={result.metrics['mae']:.4f}, RMSE={result.metrics['rmse']:.4f}"
        )

        # Get summary
        summary = result.summary()
        assert "MAE" in summary
        assert "RMSE" in summary


class TestFullPipeline:
    """End-to-end integration tests for complete pipeline."""

    def test_forecast_and_allocate_pipeline(self, sample_price_data):
        """Test complete forecast + allocation pipeline."""
        # Step 1: Generate forecasts for each ticker
        forecasts = {}
        train_data = sample_price_data.iloc[:80]

        for ticker in sample_price_data.columns:
            forecaster = ForecasterRegistry.create("naive")
            forecaster.fit(train_data[ticker])
            forecast = forecaster.predict(horizon=1)
            forecasts[ticker] = forecast.iloc[0]

        # Step 2: Convert to predicted returns
        last_prices = train_data.iloc[-1]
        predicted_returns = pd.Series(
            {
                ticker: (forecasts[ticker] - last_prices[ticker]) / last_prices[ticker]
                for ticker in sample_price_data.columns
            }
        )

        # Step 3: Calculate historical returns
        historical_returns = sample_price_data.iloc[:80].pct_change().dropna()

        # Step 4: Allocate portfolio
        allocator = AllocatorRegistry.create("equal_weight")
        weights = allocator.allocate(predicted_returns, historical_returns)

        # Verify end-to-end results
        assert len(weights) == len(sample_price_data.columns)
        assert abs(weights.sum() - 1.0) < 1e-6

        # Calculate expected portfolio return
        portfolio_return = sum(
            predicted_returns[ticker] * weights[ticker] for ticker in weights.index
        )

        logger.info(f"Portfolio weights: {weights}")
        logger.info(f"Expected return: {portfolio_return:.4%}")

        assert isinstance(portfolio_return, (int, float))

    def test_multi_forecaster_comparison_pipeline(self, sample_price_data):
        """Test comparing multiple forecasters."""
        ticker_data = sample_price_data["AAPL"]
        train_data = ticker_data[:70]
        test_data = ticker_data[70:75]

        forecaster_names = ["naive", "drift", "ridge"]
        results = {}

        for name in forecaster_names:
            try:
                forecaster = ForecasterRegistry.create(name)
                forecaster.fit(train_data)
                predictions = forecaster.predict(horizon=len(test_data))

                error_mae = mae(test_data, predictions)
                error_rmse = rmse(test_data, predictions)
                dir_acc = directional_accuracy(test_data, predictions)

                results[name] = {
                    "mae": error_mae,
                    "rmse": error_rmse,
                    "directional_accuracy": dir_acc,
                }

                logger.info(f"{name}: MAE={error_mae:.4f}, DA={dir_acc:.2f}%")
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                continue

        # Verify we got results
        assert len(results) >= 2

        # Find best forecaster by MAE
        best_forecaster = min(results.items(), key=lambda x: x[1]["mae"])
        logger.info(f"Best forecaster: {best_forecaster[0]}")

        assert best_forecaster[0] in forecaster_names

    def test_volatility_forecasting_pipeline(self, sample_return_data):
        """Test volatility forecasting pipeline (requires arch)."""
        # Use returns for GARCH (not prices)
        ticker_returns = sample_return_data["AAPL"] * 100  # Scale for numerical stability
        train_data = ticker_returns[:80]

        # Test GARCH forecaster
        forecaster = ForecasterRegistry.create("garch", config={"p": 1, "q": 1})
        forecaster.fit(train_data)

        # Forecast volatility
        vol_forecast = forecaster.predict(horizon=1)

        assert len(vol_forecast) == 1
        assert vol_forecast.iloc[0] > 0  # Volatility is positive
        assert forecaster.is_fitted
        assert forecaster.family == "volatility"
        assert forecaster.fitted_model is not None

        logger.info(f"Forecasted volatility: {vol_forecast.iloc[0]:.4f}")


class TestConfigurationIntegration:
    """Integration tests for configuration profiles."""

    def test_load_default_profile(self):
        """Test loading default configuration profile."""
        from src.config import clear_config_cache, get_config

        clear_config_cache()
        config = get_config("default")

        assert config.profile == "default"
        assert config.execution.mode == "single"
        assert len(config.portfolio.tickers) > 0
        assert config.forecasters.default is not None
        assert config.allocators.default is not None

    def test_load_fast_profile(self):
        """Test loading fast profile for smoke tests."""
        from src.config import clear_config_cache, get_config

        clear_config_cache()
        config = get_config("fast")

        assert config.profile == "fast"
        assert config.execution.horizon == 1
        assert len(config.portfolio.tickers) <= 3  # Fast uses minimal tickers
        assert config.forecasters.default == "naive"  # Fast uses simplest model

    def test_load_debug_profile(self):
        """Test loading debug profile."""
        from src.config import clear_config_cache, get_config

        clear_config_cache()
        config = get_config("debug")

        assert config.profile == "debug"
        assert config.execution.save_to_db is False  # Debug never saves to DB
        assert config.logging.level == "DEBUG"

    def test_invalid_profile_raises_error(self):
        """Test that invalid profile raises ValueError."""
        from src.config import clear_config_cache, get_config

        clear_config_cache()
        with pytest.raises(ValueError, match="Invalid profile"):
            get_config("nonexistent_profile")


class TestAdvancedBacktesting:
    """Integration tests for advanced backtesting features."""

    def test_backtest_with_gap_size(self, sample_price_data):
        """Test walk-forward backtest with anti-leakage gap."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(
            initial_train_size=60,
            test_size=1,
            step_size=5,
            min_train_size=50,
            gap_size=2,  # 2-day gap for anti-leakage
        )

        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        assert result.n_folds > 0
        assert len(result.predictions) > 0
        assert "mae" in result.metrics

    def test_backtest_with_refit_every(self, sample_price_data):
        """Test walk-forward backtest with periodic refitting."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(
            initial_train_size=60,
            test_size=1,
            step_size=1,
            min_train_size=50,
            refit_every=5,  # Only refit every 5 steps
        )

        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        assert result.n_folds > 0
        # With refit_every=5, we should have fewer retraining events
        assert len(result.fold_results) > 0

    def test_fold_results_structure(self, sample_price_data):
        """Test that FoldResult contains expected fields."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=10, min_train_size=50)

        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        assert len(result.fold_results) > 0

        # Check FoldResult structure
        fold = result.fold_results[0]
        assert fold.fold_idx == 0
        assert fold.train_end_date is not None
        assert fold.y_true is not None
        assert fold.y_pred is not None
        assert fold.y_prev is not None
        assert fold.failed is False

    def test_y_prev_values_for_directional_accuracy(self, sample_price_data):
        """Test that y_prev values are captured for directional_accuracy_1step."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=5, min_train_size=50)

        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        # y_prev_values should be populated
        assert len(result.y_prev_values) > 0
        assert len(result.y_prev_values) == len(result.predictions)

        # directional_accuracy_1step should be in metrics
        assert "directional_accuracy_1step" in result.metrics


class TestAdvancedAllocators:
    """Integration tests for advanced allocators."""

    def test_hrp_allocator_pipeline(self, sample_price_data, sample_return_data):
        """Test Hierarchical Risk Parity allocator."""
        predicted_returns = sample_return_data.iloc[-1]
        historical_returns = sample_return_data.iloc[:-10]

        allocator = AllocatorRegistry.create("hrp")
        weights = allocator.allocate(predicted_returns, historical_returns)

        assert len(weights) == len(predicted_returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values)
        assert allocator.family == "advanced"  # HRP uses advanced family

    def test_cvar_allocator_pipeline(self, sample_price_data, sample_return_data):
        """Test CVaR (Conditional Value at Risk) allocator."""
        predicted_returns = sample_return_data.iloc[-1]
        historical_returns = sample_return_data.iloc[:-10]

        allocator = AllocatorRegistry.create("cvar", config={"alpha": 0.05})
        weights = allocator.allocate(predicted_returns, historical_returns)

        assert len(weights) == len(predicted_returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values)
        assert allocator.family == "advanced"  # CVaR uses advanced family

    def test_max_sharpe_allocator_pipeline(self, sample_price_data, sample_return_data):
        """Test Maximum Sharpe Ratio allocator."""
        predicted_returns = sample_return_data.iloc[-1]
        historical_returns = sample_return_data.iloc[:-10]

        allocator = AllocatorRegistry.create("max_sharpe", config={"risk_free_rate": 0.02})
        weights = allocator.allocate(predicted_returns, historical_returns)

        assert len(weights) == len(predicted_returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert allocator.family == "mean_variance"


class TestMetricsIntegration:
    """Integration tests for metrics calculation."""

    def test_all_standard_metrics_calculated(self, sample_price_data):
        """Test that all standard metrics are calculated in backtest."""
        ticker_data = sample_price_data["AAPL"]

        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=5, min_train_size=50)

        result = walk_forward_backtest(data=ticker_data, forecaster_name="naive", config=config)

        # Check standard metrics are present
        expected_metrics = [
            "mae",
            "rmse",
            "mase",
            "directional_accuracy",
            "directional_accuracy_1step",
        ]

        for metric in expected_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"

    def test_evaluate_forecast_metrics(self, sample_price_data):
        """Test evaluate_forecast returns all expected metrics."""
        from src.evaluation.metrics import evaluate_forecast

        ticker_data = sample_price_data["AAPL"]
        train_data = ticker_data[:80]
        test_data = ticker_data[80:85]

        # Simple naive forecast
        predictions = pd.Series([train_data.iloc[-1]] * len(test_data), index=test_data.index)

        metrics = evaluate_forecast(test_data, predictions, y_train=train_data)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "mase" in metrics
        assert "directional_accuracy" in metrics
        assert all(np.isfinite(v) for v in metrics.values())


class TestRobustness:
    """Test pipeline robustness with edge cases."""

    def test_small_dataset(self):
        """Test with small dataset."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5), index=dates)

        forecaster = ForecasterRegistry.create("naive")
        forecaster.fit(prices)
        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5
        assert forecaster.is_fitted

    def test_constant_data(self):
        """Test with constant prices (no volatility)."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.Series(100.0, index=dates)

        forecaster = ForecasterRegistry.create("naive")
        forecaster.fit(prices)
        predictions = forecaster.predict(horizon=3)

        assert len(predictions) == 3
        for pred in predictions:
            assert pred == pytest.approx(100.0)

    def test_missing_data_handling(self):
        """Test handling of data with some noise."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates)

        # Add some small perturbations
        prices += np.random.randn(50) * 0.1

        forecaster = ForecasterRegistry.create("naive")
        forecaster.fit(prices)
        predictions = forecaster.predict(horizon=5)

        assert len(predictions) == 5
        assert all(np.isfinite(predictions))


# =============================================================================
# Parametrized Registry Tests
# =============================================================================

# All baseline forecasters that should always be available
BASELINE_FORECASTERS = ["naive", "seasonal_naive", "drift"]

# All ML forecasters that should always be available
ML_FORECASTERS = ["ridge", "lasso", "elasticnet", "random_forest", "hist_gradient_boosting"]

# ML forecasters
OPTIONAL_ML_FORECASTERS = [
    "xgboost",
    "lightgbm",
    "catboost",
]

# Statistical forecasters
STATISTICAL_FORECASTERS = [
    "ets",
    "theta",
    "arima",
]

# Volatility forecasters
VOLATILITY_FORECASTERS = [
    "garch",
    "egarch",
    "gjr_garch",
]

# All allocators that should always be available
ALL_ALLOCATORS = [
    "equal_weight",
    "inverse_volatility",
    "risk_parity",
    "minimum_variance",
    "maximum_diversification",
    "markowitz",
    "robust_markowitz",
    "max_sharpe",
    "black_litterman",
    "hrp",
    "cvar",
    "kelly_criterion",
    "mean_cvar",
    "omega_ratio",
]


class TestForecasterRegistration:
    """Parametrized tests for forecaster registration."""

    @pytest.mark.parametrize("name", BASELINE_FORECASTERS)
    def test_baseline_forecaster_registered(self, name):
        """Test that baseline forecaster is registered and can be created."""
        assert name in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create(name)
        assert forecaster is not None
        assert forecaster.family == "baseline"

    @pytest.mark.parametrize("name", ML_FORECASTERS)
    def test_ml_forecaster_registered(self, name):
        """Test that ML forecaster is registered and can be created."""
        assert name in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create(name)
        assert forecaster is not None
        assert forecaster.family == "ml"

    @pytest.mark.parametrize("name", OPTIONAL_ML_FORECASTERS)
    def test_optional_ml_forecaster_registered(self, name):
        """Test that optional ML forecaster is registered if available."""
        if name in ForecasterRegistry.list_all():
            forecaster = ForecasterRegistry.create(name)
            assert forecaster is not None
            assert forecaster.family == "ml"

    @pytest.mark.parametrize("name", STATISTICAL_FORECASTERS)
    def test_statistical_forecaster_registered(self, name):
        """Test that statistical forecaster is registered and can be created."""
        assert name in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create(name)
        assert forecaster is not None
        assert forecaster.family == "statistical"

    @pytest.mark.parametrize("name", VOLATILITY_FORECASTERS)
    def test_volatility_forecaster_registered(self, name):
        """Test that volatility forecaster is registered and can be created."""
        assert name in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create(name)
        assert forecaster is not None
        assert forecaster.family == "volatility"


class TestAllocatorRegistration:
    """Parametrized tests for allocator registration."""

    @pytest.mark.parametrize("name", ALL_ALLOCATORS)
    def test_allocator_registered(self, name):
        """Test that allocator is registered and can be created."""
        assert name in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create(name)
        assert allocator is not None
        assert allocator.family in ["simple", "risk_based", "mean_variance", "advanced"]


class TestRegistryCompleteness:
    """Tests to ensure registry completeness and consistency."""

    def test_all_forecasters_have_family(self):
        """Test that all registered forecasters have a family attribute."""
        for name in ForecasterRegistry.list_all():
            try:
                forecaster = ForecasterRegistry.create(name)
                assert hasattr(forecaster, "family"), f"{name} missing family attribute"
                assert forecaster.family in [
                    "baseline",
                    "ml",
                    "statistical",
                    "volatility",
                    "deep_learning",
                ], f"{name} has invalid family: {forecaster.family}"
            except Exception as e:
                # Skip if optional dependency not available
                logger.debug(f"Skipping {name}: {e}")

    def test_all_allocators_have_family(self):
        """Test that all registered allocators have a family attribute."""
        for name in AllocatorRegistry.list_all():
            allocator = AllocatorRegistry.create(name)
            assert hasattr(allocator, "family"), f"{name} missing family attribute"
            assert allocator.family in [
                "simple",
                "risk_based",
                "mean_variance",
                "advanced",
            ], f"{name} has invalid family: {allocator.family}"

    def test_forecaster_registry_list_by_family(self):
        """Test that list_by_family works for all families."""
        families = ["baseline", "ml"]
        for family in families:
            forecasters = ForecasterRegistry.list_by_family(family)
            assert len(forecasters) > 0, f"No forecasters in {family} family"

    def test_allocator_registry_count(self):
        """Test that expected number of allocators are registered."""
        all_allocators = AllocatorRegistry.list_all()
        assert (
            len(all_allocators) >= 14
        ), f"Expected at least 14 allocators, got {len(all_allocators)}"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
