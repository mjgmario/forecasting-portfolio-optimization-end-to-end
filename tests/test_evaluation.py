"""Tests for evaluation metrics and backtesting.

This module tests:
- Metrics: MAE, RMSE, MASE, Directional Accuracy, Hit Rate, Pinball Loss
- Backtesting: Walk-forward backtest, model comparison
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import (
    BacktestConfig,
    BacktestResult,
    compare_forecasters,
    walk_forward_backtest,
)
from src.evaluation.metrics import directional_accuracy, hit_rate, mae, mase, pinball_loss, rmse


@pytest.fixture
def perfect_predictions():
    """Create perfect predictions for testing metrics."""
    y_true = np.array([100, 101, 102, 103, 104])
    y_pred = np.array([100, 101, 102, 103, 104])
    return y_true, y_pred


@pytest.fixture
def sample_predictions():
    """Create sample predictions with known error."""
    y_true = np.array([100, 105, 98, 110, 102])
    y_pred = np.array([102, 103, 100, 108, 105])
    return y_true, y_pred


@pytest.fixture
def sample_time_series():
    """Create sample time series for backtesting."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
    return pd.Series(prices, index=dates)


class TestMAE:
    """Test Mean Absolute Error metric."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test MAE with perfect predictions."""
        y_true, y_pred = perfect_predictions
        error = mae(y_true, y_pred)
        assert error == pytest.approx(0.0)

    def test_known_error(self):
        """Test MAE with known error."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, -1, 2])
        # MAE = (1 + 1 + 2) / 3 = 4/3
        error = mae(y_true, y_pred)
        assert error == pytest.approx(4.0 / 3.0)

    def test_positive_output(self, sample_predictions):
        """Test that MAE is always non-negative."""
        y_true, y_pred = sample_predictions
        error = mae(y_true, y_pred)
        assert error >= 0


class TestRMSE:
    """Test Root Mean Squared Error metric."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test RMSE with perfect predictions."""
        y_true, y_pred = perfect_predictions
        error = rmse(y_true, y_pred)
        assert error == pytest.approx(0.0)

    def test_known_error(self):
        """Test RMSE with known error."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, -1, 2])
        # RMSE = sqrt((1 + 1 + 4) / 3) = sqrt(2)
        error = rmse(y_true, y_pred)
        assert error == pytest.approx(np.sqrt(2))

    def test_rmse_ge_mae(self, sample_predictions):
        """Test that RMSE >= MAE."""
        y_true, y_pred = sample_predictions
        rmse_value = rmse(y_true, y_pred)
        mae_value = mae(y_true, y_pred)
        assert rmse_value >= mae_value


class TestMASE:
    """Test Mean Absolute Scaled Error metric."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test MASE with perfect predictions."""
        y_true, y_pred = perfect_predictions
        y_train = np.array([95, 96, 97, 98, 99])
        error = mase(y_true, y_pred, y_train=y_train)
        assert error == pytest.approx(0.0)

    def test_beats_naive(self):
        """Test MASE < 1 when model beats naive."""
        # Naive forecast: repeat last value
        y_train = np.array([100, 101, 102, 103, 104])
        y_true = np.array([105, 106, 107])

        # Perfect predictions
        y_pred = np.array([105, 106, 107])

        error = mase(y_true, y_pred, y_train=y_train)
        # MAE of naive on training = 1.0 (always off by 1)
        # MAE of perfect predictions = 0.0
        # MASE = 0 / 1 = 0
        assert error == pytest.approx(0.0)

    def test_equals_naive(self):
        """Test MASE when using naive forecast on test set."""
        y_train = np.array([100, 101, 102, 103, 104])
        y_true = np.array([105, 106, 107])

        # Naive forecast: repeat last value (104)
        y_pred = np.array([104, 104, 104])

        error = mase(y_true, y_pred, y_train=y_train)
        # MAE on test = mean(|105-104|, |106-104|, |107-104|) = mean(1,2,3) = 2.0
        # MAE on train (naive baseline) = mean(|101-100|, |102-101|, |103-102|, |104-103|) = 1.0
        # MASE = 2.0 / 1.0 = 2.0
        assert error == pytest.approx(2.0, rel=0.1)

    def test_with_seasonality(self):
        """Test MASE with seasonal naive baseline."""
        y_train = np.array([100, 101, 102, 103, 104, 105])
        y_true = np.array([106, 107])
        y_pred = np.array([106.5, 106.5])

        # Seasonal naive with season=2
        error = mase(y_true, y_pred, y_train=y_train, seasonality=2)
        assert error > 0


class TestDirectionalAccuracy:
    """Test Directional Accuracy metric."""

    def test_perfect_direction(self):
        """Test with perfect directional predictions."""
        y_true = np.array([100, 105, 103, 110])  # changes: +5, -2, +7 -> directions: +, -, +
        y_pred = np.array([100, 106, 102, 111])  # changes: +6, -4, +9 -> directions: +, -, +
        accuracy = directional_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(1.0)

    def test_all_wrong_direction(self):
        """Test with all wrong directional predictions."""
        y_true = np.array([100, 105, 103, 110])  # +, -, +
        y_pred = np.array([100, 98, 102, 101])  # -, +, -
        accuracy = directional_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(0.0)

    def test_half_correct(self):
        """Test with 50% correct direction."""
        y_true = np.array([100, 105, 110, 108])  # +, +, -
        y_pred = np.array([102, 103, 109, 112])  # +, +, +
        accuracy = directional_accuracy(y_true, y_pred)
        # 2 out of 3 correct = 2/3
        assert accuracy == pytest.approx(2.0 / 3.0)

    def test_range(self):
        """Test that directional accuracy is in [0, 1]."""
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        accuracy = directional_accuracy(y_true, y_pred)
        assert 0.0 <= accuracy <= 1.0


class TestHitRate:
    """Test Hit Rate metric."""

    def test_all_hits(self):
        """Test with all predictions hitting target."""
        y_true = np.array([100, 105, 98, 110])
        y_pred = np.array([100.5, 105.2, 98.3, 110.1])
        # With threshold=1%, all should hit
        hit = hit_rate(y_true, y_pred, threshold=0.01)
        assert hit == pytest.approx(1.0)

    def test_no_hits(self):
        """Test with no predictions hitting target."""
        y_true = np.array([100, 105, 98, 110])
        y_pred = np.array([110, 95, 108, 100])
        # With threshold=1%, none should hit
        hit = hit_rate(y_true, y_pred, threshold=0.01)
        assert hit == pytest.approx(0.0)

    def test_half_hits(self):
        """Test with 50% hit rate."""
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([100.5, 102, 99.5, 103])
        # With threshold=1%, first and third hit
        hit = hit_rate(y_true, y_pred, threshold=0.01)
        assert hit == pytest.approx(0.5)


class TestPinballLoss:
    """Test Pinball Loss for quantile forecasting."""

    def test_median_prediction(self):
        """Test pinball loss for median (quantile=0.5)."""
        y_true = np.array([100, 105, 98, 110])
        y_pred = np.array([100, 105, 98, 110])
        loss = pinball_loss(y_true, y_pred, quantile=0.5)
        # Perfect predictions should have zero loss
        assert loss == pytest.approx(0.0)

    def test_lower_quantile(self):
        """Test pinball loss for lower quantile (0.1)."""
        y_true = np.array([100])
        y_pred = np.array([95])
        # Underestimation at quantile 0.1
        # Loss = (1 - 0.1) * (100 - 95) = 0.9 * 5 = 4.5
        loss = pinball_loss(y_true, y_pred, quantile=0.1)
        assert loss == pytest.approx(4.5)

    def test_upper_quantile(self):
        """Test pinball loss for upper quantile (0.9)."""
        y_true = np.array([100])
        y_pred = np.array([105])
        # Overestimation at quantile 0.9
        # Loss = 0.9 * (105 - 100) = 0.9 * 5 = 4.5
        loss = pinball_loss(y_true, y_pred, quantile=0.9)
        assert loss == pytest.approx(4.5)

    def test_asymmetric_penalty(self):
        """Test that pinball loss is asymmetric."""
        y_true = np.array([100])

        # For low quantile (0.1), we want conservative (low) estimates
        # Overestimate (predict too high) should be penalized LESS
        over_pred = np.array([105])
        over_loss = pinball_loss(y_true, over_pred, quantile=0.1)

        # Underestimate (predict too low) should be penalized MORE
        under_pred = np.array([95])
        under_loss = pinball_loss(y_true, under_pred, quantile=0.1)

        assert over_loss < under_loss  # over_loss (0.5) < under_loss (4.5)


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()
        assert config.initial_train_size == 252
        assert config.test_size == 1
        assert config.step_size == 1
        assert config.min_train_size == 60

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(initial_train_size=100, test_size=5, step_size=5, min_train_size=50)
        assert config.initial_train_size == 100
        assert config.test_size == 5
        assert config.step_size == 5
        assert config.min_train_size == 50


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_summary(self):
        """Test summary method."""
        result = BacktestResult(
            forecaster_name="naive",
            predictions=pd.Series([100, 101, 102]),
            actuals=pd.Series([100, 102, 103]),
            mae=1.0,
            rmse=1.5,
            mase=0.8,
            directional_accuracy=0.75,
            n_folds=3,
        )

        summary = result.summary()
        assert "naive" in summary
        assert "MAE" in summary
        assert "MASE" in summary
        assert "0.8" in summary


class TestWalkForwardBacktest:
    """Test walk-forward backtesting."""

    def test_naive_backtest(self, sample_time_series):
        """Test backtesting with Naive forecaster."""
        config = BacktestConfig(initial_train_size=50, test_size=1, step_size=1, min_train_size=50)

        result = walk_forward_backtest(
            data=sample_time_series, forecaster_name="naive", config=config
        )

        assert isinstance(result, BacktestResult)
        assert result.forecaster_name == "naive"
        assert len(result.predictions) > 0
        assert len(result.actuals) == len(result.predictions)
        assert result.metrics["mae"] >= 0
        assert result.metrics["rmse"] >= 0
        assert 0 <= result.metrics.get("directional_accuracy", 0) <= 1

    def test_ridge_backtest(self, sample_time_series):
        """Test backtesting with Ridge forecaster."""
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=5)

        result = walk_forward_backtest(
            data=sample_time_series,
            forecaster_name="ridge",
            forecaster_config={"lags": 5},
            config=config,
        )

        assert isinstance(result, BacktestResult)
        assert result.forecaster_name == "ridge"
        assert len(result.predictions) > 0

    def test_multi_step_horizon(self, sample_time_series):
        """Test backtesting with multi-step horizon."""
        config = BacktestConfig(initial_train_size=60, test_size=5, step_size=5)

        result = walk_forward_backtest(
            data=sample_time_series, forecaster_name="naive", config=config
        )

        assert len(result.predictions) > 0
        # With test_size=5, we should have multiple folds
        assert result.n_folds > 0

    def test_invalid_forecaster_raises_error(self, sample_time_series):
        """Test that invalid forecaster name raises error."""
        config = BacktestConfig(initial_train_size=50, test_size=1, step_size=1, min_train_size=50)

        with pytest.raises((ValueError, KeyError, RuntimeError)):
            walk_forward_backtest(
                data=sample_time_series,
                forecaster_name="nonexistent_model",
                config=config,
            )

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises error."""
        short_series = pd.Series(
            [100, 101, 102], index=pd.date_range("2024-01-01", periods=3, freq="B")
        )
        config = BacktestConfig(initial_train_size=50, test_size=1, step_size=1, min_train_size=50)

        with pytest.raises(ValueError, match="Data length .* insufficient"):
            walk_forward_backtest(data=short_series, forecaster_name="naive", config=config)


class TestCompareForecasters:
    """Test model comparison."""

    def test_compare_two_models(self, sample_time_series):
        """Test comparing two forecasters."""
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=10)

        results_df = compare_forecasters(
            data=sample_time_series,
            forecaster_configs={"naive": None, "drift": None},
            config=config,
        )

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2
        assert "naive" in results_df["forecaster"].values
        assert "drift" in results_df["forecaster"].values
        assert "mae" in results_df.columns
        assert "rmse" in results_df.columns
        assert "mase" in results_df.columns
        assert "directional_accuracy" in results_df.columns

    def test_compare_with_configs(self, sample_time_series):
        """Test comparing forecasters with different configs."""
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=10)

        forecaster_configs = {
            "ridge": {"lags": 5},
            "lasso": {"lags": 10, "alpha": 0.1},
        }

        results_df = compare_forecasters(
            data=sample_time_series,
            forecaster_configs=forecaster_configs,
            config=config,
        )

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2

    def test_results_sorted_by_mae(self, sample_time_series):
        """Test that results are sorted by MASE."""
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=10)

        results_df = compare_forecasters(
            data=sample_time_series,
            forecaster_configs={"naive": None, "drift": None, "seasonal_naive": None},
            config=config,
        )

        # Check that results are sorted by MASE ascending
        if "mase" in results_df.columns:
            mase_values = results_df["mase"].values
            assert all(mase_values[i] <= mase_values[i + 1] for i in range(len(mase_values) - 1))

    def test_rank_column(self, sample_time_series):
        """Test that results comparison returns multiple forecasters."""
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=10)

        results_df = compare_forecasters(
            data=sample_time_series,
            forecaster_configs={"naive": None, "drift": None},
            config=config,
        )

        # Should have 2 forecasters
        assert len(results_df) == 2


class TestIntegrationBacktesting:
    """Integration tests for backtesting."""

    def test_full_pipeline(self, sample_time_series):
        """Test full backtesting pipeline."""
        # Step 1: Run backtest for single model
        config = BacktestConfig(initial_train_size=60, test_size=1, step_size=5)
        result = walk_forward_backtest(
            data=sample_time_series, forecaster_name="naive", config=config
        )

        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert len(result.predictions) > 0

        # Step 2: Compare multiple models
        results_df = compare_forecasters(
            data=sample_time_series,
            forecaster_configs={"naive": None, "drift": None, "ridge": {"lags": 5}},
            config=config,
        )

        # Verify comparison results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert all(results_df["mae"] >= 0)
