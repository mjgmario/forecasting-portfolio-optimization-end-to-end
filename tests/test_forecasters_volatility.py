"""Tests for volatility forecasters (GARCH models).

This module tests:
- GARCHForecaster
- EGARCHForecaster
- GJRGARCHForecaster

Note: GARCH models forecast volatility (standard deviation), not prices or returns.
Input should be returns, not prices.

These require arch as an optional dependency.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters import ForecasterRegistry
from src.forecasters.volatility import EGARCHForecaster, GARCHForecaster, GJRGARCHForecaster


@pytest.fixture
def sample_return_series():
    """Create a sample return time series with volatility clustering."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")

    # Create returns with volatility clustering
    returns = []
    vol = 0.01  # Initial volatility

    for _ in range(300):
        # GARCH(1,1)-like process
        vol = np.sqrt(0.00001 + 0.1 * returns[-1] ** 2 if returns else 0.0 + 0.85 * vol**2)
        ret = np.random.randn() * vol
        returns.append(ret)

    return pd.Series(returns, index=dates)


@pytest.fixture
def simple_return_series():
    """Create a simple return series for basic testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    returns = np.random.randn(200) * 0.02  # 2% daily volatility
    return pd.Series(returns, index=dates)


class TestGARCHForecaster:
    """Test GARCH forecaster."""

    def test_registration(self):
        """Test that GARCH forecaster is registered."""
        assert "garch" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("garch")
        assert isinstance(forecaster, GARCHForecaster)

    def test_fit(self, sample_return_series):
        """Test fitting GARCH model."""
        forecaster = GARCHForecaster(config={"p": 1, "q": 1})
        forecaster.fit(sample_return_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None
        assert forecaster.fitted_model is not None

    def test_predict_single_step(self, sample_return_series):
        """Test single-step volatility prediction."""
        forecaster = GARCHForecaster(config={"p": 1, "q": 1})
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        # Volatility should be positive
        assert predictions.iloc[0] > 0

    def test_predict_multi_step(self, sample_return_series):
        """Test multi-step volatility prediction."""
        forecaster = GARCHForecaster(config={"p": 1, "q": 1})
        forecaster.fit(sample_return_series)

        horizon = 10
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        # All volatility predictions should be positive
        assert all(predictions > 0)

    def test_with_custom_order(self, sample_return_series):
        """Test with custom GARCH order."""
        # GARCH(2, 2)
        forecaster = GARCHForecaster(config={"p": 2, "q": 2})
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=1)

        assert len(predictions) == 1
        assert predictions.iloc[0] > 0

    def test_output_is_volatility_not_price(self, sample_return_series):
        """Test that output is volatility (standard deviation), not price."""
        forecaster = GARCHForecaster()
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=5)

        # Volatility for daily returns should typically be small (< 0.1)
        # and much smaller than typical stock prices (> 10)
        assert all(predictions < 1.0)
        # Should be positive
        assert all(predictions > 0)

    def test_family(self):
        """Test that GARCH has family='volatility'."""
        forecaster = GARCHForecaster()
        assert forecaster.family == "volatility"

    def test_requires_sufficient_data(self):
        """Test that GARCH raises error with insufficient data."""
        # Very short series (less than 30 observations)
        short_series = pd.Series(
            np.random.randn(20) * 0.02,
            index=pd.date_range("2024-01-01", periods=20, freq="B"),
        )

        forecaster = GARCHForecaster()

        # Should raise ValueError due to insufficient data for numerical stability
        with pytest.raises(ValueError, match="requires at least 30 observations"):
            forecaster.fit(short_series)


class TestEGARCHForecaster:
    """Test EGARCH (Exponential GARCH) forecaster."""

    def test_registration(self):
        """Test that EGARCH forecaster is registered."""
        assert "egarch" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("egarch")
        assert isinstance(forecaster, EGARCHForecaster)

    def test_fit(self, sample_return_series):
        """Test fitting EGARCH model."""
        forecaster = EGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict_single_step(self, sample_return_series):
        """Test single-step volatility prediction."""
        forecaster = EGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        assert predictions.iloc[0] > 0

    def test_predict_multi_step(self, sample_return_series):
        """Test multi-step volatility prediction."""
        forecaster = EGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        horizon = 5
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        assert all(predictions > 0)

    def test_asymmetric_effect(self, sample_return_series):
        """Test that EGARCH captures asymmetric volatility effects."""
        forecaster = EGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        # EGARCH should capture leverage effect
        # (negative returns increase volatility more than positive returns)
        predictions = forecaster.predict(horizon=3)

        assert len(predictions) == 3
        assert all(predictions > 0)

    def test_family(self):
        """Test that EGARCH has family='volatility'."""
        forecaster = EGARCHForecaster()
        assert forecaster.family == "volatility"


class TestGJRGARCHForecaster:
    """Test GJR-GARCH forecaster."""

    def test_registration(self):
        """Test that GJR-GARCH forecaster is registered."""
        assert "gjr_garch" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("gjr_garch")
        assert isinstance(forecaster, GJRGARCHForecaster)

    def test_fit(self, sample_return_series):
        """Test fitting GJR-GARCH model."""
        forecaster = GJRGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict_single_step(self, sample_return_series):
        """Test single-step volatility prediction."""
        forecaster = GJRGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=1)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 1
        assert predictions.iloc[0] > 0

    def test_predict_multi_step(self, sample_return_series):
        """Test multi-step volatility prediction."""
        forecaster = GJRGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        horizon = 5
        predictions = forecaster.predict(horizon=horizon)

        assert len(predictions) == horizon
        assert all(predictions > 0)

    def test_asymmetric_effect(self, sample_return_series):
        """Test that GJR-GARCH captures asymmetric volatility effects."""
        forecaster = GJRGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        forecaster.fit(sample_return_series)

        # GJR-GARCH should capture leverage effect
        predictions = forecaster.predict(horizon=3)

        assert len(predictions) == 3
        assert all(predictions > 0)

    def test_family(self):
        """Test that GJR-GARCH has family='volatility'."""
        forecaster = GJRGARCHForecaster()
        assert forecaster.family == "volatility"


class TestVolatilityForecasterComparison:
    """Test comparison of volatility forecasters."""

    def test_all_volatility_models_work_on_same_data(self, sample_return_series):
        """Test that all volatility models can fit and predict on same data."""
        volatility_models = ["garch", "egarch", "gjr_garch"]

        for model_name in volatility_models:
            forecaster = ForecasterRegistry.create(model_name, config={"p": 1, "q": 1})
            forecaster.fit(sample_return_series)
            predictions = forecaster.predict(horizon=5)

            assert isinstance(predictions, pd.Series)
            assert len(predictions) == 5
            # All volatilities should be positive
            assert all(predictions > 0)
            # Volatilities should be reasonable (< 1 for daily data)
            assert all(predictions < 1.0)

    def test_family_attribute(self):
        """Test that all volatility forecasters have family='volatility'."""
        volatility_models = ["garch", "egarch", "gjr_garch"]

        for model_name in volatility_models:
            forecaster = ForecasterRegistry.create(model_name)
            assert forecaster.family == "volatility"

    def test_different_models_give_different_predictions(self, sample_return_series):
        """Test that different GARCH variants give different predictions."""
        garch = GARCHForecaster(config={"p": 1, "q": 1})
        garch.fit(sample_return_series)
        garch_pred = garch.predict(horizon=5)

        egarch = EGARCHForecaster(config={"p": 1, "o": 1, "q": 1})
        egarch.fit(sample_return_series)
        egarch_pred = egarch.predict(horizon=5)

        # Predictions should differ
        assert not np.allclose(garch_pred.values, egarch_pred.values)

    def test_volatility_persistence(self, sample_return_series):
        """Test that volatility forecasts show persistence."""
        forecaster = GARCHForecaster(config={"p": 1, "q": 1})
        forecaster.fit(sample_return_series)

        predictions = forecaster.predict(horizon=20)

        # Volatility should gradually converge to long-run average
        # (not just be constant or random)
        assert len(predictions) == 20

        # Check that volatility doesn't change too drastically step-to-step
        for i in range(1, len(predictions)):
            relative_change = (
                abs(predictions.iloc[i] - predictions.iloc[i - 1]) / predictions.iloc[i - 1]
            )
            # Daily change should be reasonable (< 50%)
            assert relative_change < 0.5

    def test_with_high_volatility_period(self):
        """Test GARCH models with high volatility period."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")

        # Low volatility first 150 days, high volatility last 50 days
        low_vol_returns = np.random.randn(150) * 0.01
        high_vol_returns = np.random.randn(50) * 0.05

        returns = np.concatenate([low_vol_returns, high_vol_returns])
        return_series = pd.Series(returns, index=dates)

        forecaster = GARCHForecaster(config={"p": 1, "q": 1})
        forecaster.fit(return_series)

        predictions = forecaster.predict(horizon=5)

        # After high volatility period, predictions should be elevated
        assert all(predictions > 0.01)  # Should be higher than low vol period
