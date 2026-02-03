"""Tests for main.py module.

Tests cover:
- Helper functions for data extraction and preprocessing
- Forecaster selection logic
- Portfolio optimization workflow
- Single forecast execution
- Model comparison functionality
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.main import (
    ValidationError,
    _build_historical_returns,
    _extract_and_preprocess,
    _forecast_individual_mode,
    _optimize_portfolio_weights,
    _run_allocators_for_forecaster,
    _run_forecaster_for_all_tickers,
    _select_forecaster_for_run,
    prepare_combined_training_data,
    run_single_forecast,
    validate_allocator,
    validate_forecaster,
    validate_horizon,
    validate_tickers,
)


class TestValidateHorizon:
    """Tests for validate_horizon function."""

    def test_valid_horizon(self):
        """Test valid horizons pass validation."""
        validate_horizon(1)
        validate_horizon(21)
        validate_horizon(252)
        validate_horizon(365)

    def test_horizon_too_small(self):
        """Test horizon less than 1 raises error."""
        with pytest.raises(ValidationError, match="at least 1"):
            validate_horizon(0)
        with pytest.raises(ValidationError, match="at least 1"):
            validate_horizon(-5)

    def test_horizon_too_large(self):
        """Test horizon exceeding MAX_HORIZON raises error."""
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_horizon(366)
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_horizon(1000)


class TestValidateForecaster:
    """Tests for validate_forecaster function."""

    def test_valid_forecaster(self):
        """Test valid forecaster names pass validation."""
        validate_forecaster("naive")
        validate_forecaster("drift")
        validate_forecaster("ridge")

    def test_invalid_forecaster(self):
        """Test invalid forecaster name raises error."""
        with pytest.raises(ValidationError, match="Unknown forecaster"):
            validate_forecaster("nonexistent_model")
        with pytest.raises(ValidationError, match="Unknown forecaster"):
            validate_forecaster("")


class TestValidateAllocator:
    """Tests for validate_allocator function."""

    def test_valid_allocator(self):
        """Test valid allocator names pass validation."""
        validate_allocator("equal_weight")
        validate_allocator("inverse_volatility")
        validate_allocator("markowitz")

    def test_invalid_allocator(self):
        """Test invalid allocator name raises error."""
        with pytest.raises(ValidationError, match="Unknown allocator"):
            validate_allocator("nonexistent_allocator")
        with pytest.raises(ValidationError, match="Unknown allocator"):
            validate_allocator("")


class TestValidateTickers:
    """Tests for validate_tickers function."""

    def test_valid_tickers(self):
        """Test valid ticker lists pass validation."""
        validate_tickers(["AAPL"])
        validate_tickers(["AAPL", "MSFT", "GOOGL"])
        validate_tickers(["BRK.B"])  # Dots allowed
        validate_tickers(["BRK-A"])  # Hyphens allowed

    def test_empty_tickers(self):
        """Test empty ticker list raises error."""
        with pytest.raises(ValidationError, match="At least one ticker"):
            validate_tickers([])

    def test_invalid_ticker_format(self):
        """Test invalid ticker format raises error."""
        with pytest.raises(ValidationError, match="Invalid ticker"):
            validate_tickers(["AAPL", ""])
        with pytest.raises(ValidationError, match="Invalid ticker"):
            validate_tickers(["AAPL", "INVALID@TICKER"])


@pytest.fixture
def mock_portfolio_data():
    """Create mock portfolio data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)

    return {
        "AAPL": pd.DataFrame(
            {"Price": 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)}, index=dates
        ),
        "MSFT": pd.DataFrame(
            {"Price": 150 + np.cumsum(np.random.randn(100) * 0.6 + 0.15)}, index=dates
        ),
        "GOOGL": pd.DataFrame(
            {"Price": 200 + np.cumsum(np.random.randn(100) * 0.7 + 0.12)}, index=dates
        ),
    }


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = MagicMock()
    config.portfolio.tickers = ["AAPL", "MSFT", "GOOGL"]
    config.forecasters.default = "naive"
    config.forecasters.mode = "single"
    config.forecasters.training_mode = "individual"
    config.forecasters.candidates = ["naive", "drift"]
    config.forecasters.selection_metric = "directional_accuracy"
    config.forecasters.selection_backtest_steps = 10
    config.forecasters.skip = []
    config.forecasters.get_forecaster_for_ticker = MagicMock(return_value="naive")
    config.allocators.default = "equal_weight"
    config.dates.start_date = "2024-01-01"
    config.dates.effective_end_date = "2024-06-01"
    config.backtesting.initial_train_size = 60
    config.backtesting.test_size = 1
    config.backtesting.min_train_size = 30
    config.backtesting.comparison = {"step_size": 5}
    config.should_skip_forecaster = MagicMock(return_value=False)
    config.get_forecaster_config = MagicMock(return_value={})
    config.get_allocator_config = MagicMock(return_value={})
    return config


class TestBuildHistoricalReturns:
    """Tests for _build_historical_returns function."""

    def test_builds_returns_from_dataframe(self, mock_portfolio_data):
        """Test building returns from DataFrame portfolio data."""
        tickers = ["AAPL", "MSFT"]
        returns = _build_historical_returns(mock_portfolio_data, tickers)

        assert isinstance(returns, pd.DataFrame)
        assert "AAPL" in returns.columns
        assert "MSFT" in returns.columns
        assert len(returns) == 99  # One less due to pct_change
        assert not returns.isna().any().any()

    def test_handles_missing_tickers(self, mock_portfolio_data):
        """Test handling of tickers not in data."""
        tickers = ["AAPL", "UNKNOWN"]
        returns = _build_historical_returns(mock_portfolio_data, tickers)

        assert "AAPL" in returns.columns
        assert "UNKNOWN" not in returns.columns

    def test_handles_series_data(self):
        """Test handling of Series portfolio data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        portfolio_data = {
            "AAPL": pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates)
        }
        tickers = ["AAPL"]
        returns = _build_historical_returns(portfolio_data, tickers)

        assert isinstance(returns, pd.DataFrame)
        assert "AAPL" in returns.columns


class TestPrepareCombinedTrainingData:
    """Tests for prepare_combined_training_data function."""

    def test_combines_multiple_tickers(self, mock_portfolio_data):
        """Test combining data from multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        combined = prepare_combined_training_data(mock_portfolio_data, tickers)

        assert isinstance(combined, pd.DataFrame)
        assert "ticker" in combined.columns
        assert "ticker_idx" in combined.columns
        assert set(combined["ticker"].unique()) == set(tickers)

    def test_handles_missing_tickers(self, mock_portfolio_data):
        """Test handling of missing tickers."""
        tickers = ["AAPL", "UNKNOWN"]
        combined = prepare_combined_training_data(mock_portfolio_data, tickers)

        assert "AAPL" in combined["ticker"].values
        assert "UNKNOWN" not in combined["ticker"].values

    def test_returns_empty_for_no_data(self):
        """Test returns empty DataFrame when no data available."""
        combined = prepare_combined_training_data({}, ["AAPL", "MSFT"])
        assert combined.empty

    def test_handles_series_data(self):
        """Test handling of Series instead of DataFrame."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        portfolio_data = {"AAPL": pd.Series(100 + np.random.randn(50).cumsum(), index=dates)}
        combined = prepare_combined_training_data(portfolio_data, ["AAPL"])

        assert not combined.empty
        assert "Price" in combined.columns


class TestSelectForecasterForRun:
    """Tests for _select_forecaster_for_run function."""

    def test_returns_specified_forecaster(self, mock_portfolio_data, mock_config):
        """Test returns user-specified forecaster."""
        result = _select_forecaster_for_run(mock_portfolio_data, ["AAPL"], "xgboost", mock_config)
        assert result == "xgboost"

    def test_returns_default_in_single_mode(self, mock_portfolio_data, mock_config):
        """Test returns default forecaster in single mode."""
        mock_config.forecasters.mode = "single"
        result = _select_forecaster_for_run(mock_portfolio_data, ["AAPL"], None, mock_config)
        assert result == "naive"

    def test_auto_mode_with_missing_data(self, mock_config):
        """Test auto mode falls back to default when data missing."""
        mock_config.forecasters.mode = "auto"
        result = _select_forecaster_for_run({}, ["AAPL"], None, mock_config)
        assert result == "naive"


class TestRunForecasterForAllTickers:
    """Tests for _run_forecaster_for_all_tickers function."""

    def test_generates_predictions(self, mock_portfolio_data, mock_config):
        """Test generates predictions for all tickers."""
        predictions, returns = _run_forecaster_for_all_tickers(
            "naive", mock_portfolio_data, ["AAPL", "MSFT"], horizon=5, config=mock_config
        )

        assert "AAPL" in predictions
        assert "MSFT" in predictions
        assert "AAPL" in returns
        assert "MSFT" in returns

    def test_handles_forecaster_failure(self, mock_portfolio_data, mock_config):
        """Test handles forecaster failure gracefully."""
        # Use an empty portfolio to cause failure
        predictions, returns = _run_forecaster_for_all_tickers(
            "naive", {}, ["UNKNOWN"], horizon=5, config=mock_config
        )

        assert len(predictions) == 0
        assert len(returns) == 0

    def test_skips_missing_tickers(self, mock_portfolio_data, mock_config):
        """Test skips tickers not in data."""
        predictions, returns = _run_forecaster_for_all_tickers(
            "naive",
            mock_portfolio_data,
            ["AAPL", "UNKNOWN", "MSFT"],
            horizon=5,
            config=mock_config,
        )

        assert "AAPL" in predictions
        assert "MSFT" in predictions
        assert "UNKNOWN" not in predictions


class TestRunAllocatorsForForecaster:
    """Tests for _run_allocators_for_forecaster function."""

    def test_runs_allocators(self, mock_portfolio_data, mock_config):
        """Test runs allocators and returns results."""
        historical_returns = _build_historical_returns(
            mock_portfolio_data, ["AAPL", "MSFT", "GOOGL"]
        )
        predicted_returns = {"AAPL": 0.05, "MSFT": 0.03, "GOOGL": 0.02}

        results = _run_allocators_for_forecaster(
            "naive",
            predicted_returns,
            historical_returns,
            ["equal_weight", "inverse_volatility"],
            mock_config,
        )

        assert len(results) == 2
        assert all("weights" in r for r in results)
        assert all("top_ticker" in r for r in results)

    def test_handles_allocator_failure(self, mock_portfolio_data, mock_config):
        """Test handles allocator failure gracefully."""
        historical_returns = _build_historical_returns(mock_portfolio_data, ["AAPL"])
        predicted_returns = {"AAPL": 0.05}

        # Should not raise, just skip failed allocators
        results = _run_allocators_for_forecaster(
            "naive",
            predicted_returns,
            historical_returns,
            ["nonexistent_allocator"],
            mock_config,
        )

        # Empty results because allocator doesn't exist
        assert isinstance(results, list)


class TestOptimizePortfolioWeights:
    """Tests for _optimize_portfolio_weights function."""

    def test_optimizes_weights(self, mock_portfolio_data):
        """Test optimizes portfolio weights."""
        predicted_returns = {"AAPL": 0.05, "MSFT": 0.03, "GOOGL": 0.02}

        weights = _optimize_portfolio_weights(
            predicted_returns, mock_portfolio_data, "equal_weight", None
        )

        assert isinstance(weights, (dict, pd.Series))
        # Handle both dict and Series
        weight_sum = weights.sum() if hasattr(weights, "sum") else sum(weights.values())
        assert weight_sum == pytest.approx(1.0)

    def test_raises_on_invalid_allocator(self, mock_portfolio_data):
        """Test raises error with invalid allocator."""
        predicted_returns = {"AAPL": 0.05}

        with pytest.raises((KeyError, ValueError)):
            _optimize_portfolio_weights(predicted_returns, mock_portfolio_data, "nonexistent", None)


class TestForecastIndividualMode:
    """Tests for _forecast_individual_mode function."""

    def test_forecasts_for_all_tickers(self, mock_portfolio_data, mock_config):
        """Test generates forecasts for all tickers."""
        predictions, returns = _forecast_individual_mode(
            mock_portfolio_data, ["AAPL", "MSFT"], "naive", None, mock_config
        )

        assert "AAPL" in predictions
        assert "MSFT" in predictions
        assert len(predictions) == 2

    def test_uses_override_forecaster(self, mock_portfolio_data, mock_config):
        """Test uses override forecaster when specified."""
        predictions, returns = _forecast_individual_mode(
            mock_portfolio_data, ["AAPL"], "naive", "drift", mock_config
        )

        assert "AAPL" in predictions


class TestRunSingleForecast:
    """Tests for run_single_forecast function."""

    @patch("src.main.extract_data")
    @patch("src.main.preprocess_data")
    @patch("src.main.get_db")
    def test_runs_with_mocked_data(self, mock_db, mock_preprocess, mock_extract, mock_config):
        """Test full run with mocked data extraction."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)

        # Mock extract_data to return raw data
        mock_extract.return_value = {
            "AAPL": pd.DataFrame(
                {
                    "Open": np.random.randn(100) + 100,
                    "High": np.random.randn(100) + 101,
                    "Low": np.random.randn(100) + 99,
                    "Close": 100 + np.cumsum(np.random.randn(100) * 0.5),
                    "Volume": np.random.randint(1000000, 5000000, 100),
                },
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Open": np.random.randn(100) + 150,
                    "High": np.random.randn(100) + 151,
                    "Low": np.random.randn(100) + 149,
                    "Close": 150 + np.cumsum(np.random.randn(100) * 0.6),
                    "Volume": np.random.randint(1000000, 5000000, 100),
                },
                index=dates,
            ),
        }

        # Mock preprocess_data to return data with Price column
        np.random.seed(42)
        mock_preprocess.return_value = {
            "AAPL": pd.DataFrame(
                {"Price": 100 + np.cumsum(np.random.randn(100) * 0.5)}, index=dates
            ),
            "MSFT": pd.DataFrame(
                {"Price": 150 + np.cumsum(np.random.randn(100) * 0.6)}, index=dates
            ),
        }

        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        with patch("src.main.get_config", return_value=mock_config):
            results = run_single_forecast(
                tickers=["AAPL", "MSFT"],
                forecaster_name="naive",
                allocator_name="equal_weight",
                save_to_db=False,
                config=mock_config,
            )

        assert "predictions" in results
        assert "weights" in results
        assert results["forecaster"] == "naive"
        assert results["allocator"] == "equal_weight"

    def test_returns_empty_on_no_data(self, mock_config):
        """Test returns empty dict when no data extracted."""
        with (
            patch("src.main.extract_data", return_value={}),
            patch("src.main.get_config", return_value=mock_config),
        ):
            results = run_single_forecast(tickers=["AAPL"], save_to_db=False, config=mock_config)

        assert results == {}


class TestExtractAndPreprocess:
    """Tests for _extract_and_preprocess function."""

    @patch("src.main.extract_data")
    def test_extracts_and_preprocesses(self, mock_extract):
        """Test extraction and preprocessing pipeline."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        mock_extract.return_value = {
            "AAPL": pd.DataFrame(
                {
                    "Open": np.random.randn(50) + 100,
                    "High": np.random.randn(50) + 101,
                    "Low": np.random.randn(50) + 99,
                    "Close": 100 + np.cumsum(np.random.randn(50) * 0.5),
                    "Volume": np.random.randint(1000000, 5000000, 50),
                },
                index=dates,
            )
        }

        result = _extract_and_preprocess(["AAPL"], "2024-01-01", "2024-03-01")

        assert "AAPL" in result
        mock_extract.assert_called_once()

    @patch("src.main.extract_data")
    def test_raises_on_no_data(self, mock_extract):
        """Test raises ValueError when no data extracted."""
        mock_extract.return_value = {}

        with pytest.raises(ValueError, match="No data extracted"):
            _extract_and_preprocess(["UNKNOWN"], "2024-01-01", "2024-03-01")
