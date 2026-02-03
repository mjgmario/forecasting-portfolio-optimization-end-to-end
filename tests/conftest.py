"""Pytest configuration and shared fixtures.

This module provides:
- Common fixtures for all test modules
- Test configuration
- Utility functions for testing
"""

import warnings

import numpy as np
import pandas as pd
import pytest


# Configure numpy and pandas for testing
@pytest.fixture(autouse=True)
def configure_numpy_pandas():
    """Configure numpy and pandas settings for tests."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure pandas
    pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

    # Suppress specific warnings during tests
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Prophet.*")
    warnings.filterwarnings("ignore", message=".*statsforecast.*")


# Common fixtures that can be used across all test modules
@pytest.fixture
def random_seed():
    """Fixture to reset random seed."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_dates():
    """Generate sample business day dates."""
    return pd.date_range("2024-01-01", periods=100, freq="B")


@pytest.fixture
def sample_prices(sample_dates):
    """Generate sample price series."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(sample_dates)) * 0.5 + 0.1)
    return pd.Series(prices, index=sample_dates)


@pytest.fixture
def sample_returns(sample_dates):
    """Generate sample return series."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.02
    return pd.Series(returns, index=sample_dates)


@pytest.fixture
def multi_asset_returns(sample_dates):
    """Generate multi-asset return series."""
    np.random.seed(42)
    n_assets = 4
    returns_dict = {}

    for i in range(n_assets):
        returns = np.random.randn(len(sample_dates)) * 0.02 + 0.001
        returns_dict[f"ASSET{i+1}"] = returns

    return pd.DataFrame(returns_dict, index=sample_dates)


# =============================================================================
# Edge Case Fixtures
# =============================================================================


@pytest.fixture
def returns_with_nan(sample_dates):
    """Generate returns series with NaN values."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.02
    returns[10] = np.nan
    returns[50] = np.nan
    returns[90] = np.nan
    return pd.Series(returns, index=sample_dates)


@pytest.fixture
def returns_with_inf(sample_dates):
    """Generate returns series with infinite values."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.02
    returns[25] = np.inf
    returns[75] = -np.inf
    return pd.Series(returns, index=sample_dates)


@pytest.fixture
def returns_with_outliers(sample_dates):
    """Generate returns series with extreme outliers."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.02
    returns[20] = 0.5  # +50% return (extreme positive)
    returns[40] = -0.4  # -40% return (extreme negative)
    returns[60] = 1.0  # +100% return (black swan)
    return pd.Series(returns, index=sample_dates)


@pytest.fixture
def constant_returns(sample_dates):
    """Generate constant returns (zero variance)."""
    return pd.Series([0.001] * len(sample_dates), index=sample_dates)


@pytest.fixture
def minimal_data():
    """Generate minimal training data (5 points)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    np.random.seed(42)
    return pd.Series([100, 101, 102, 101.5, 103], index=dates)


@pytest.fixture
def single_point_data():
    """Generate single point training data."""
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    return pd.Series([100.0], index=dates)


@pytest.fixture
def single_asset_portfolio():
    """Generate single asset portfolio data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02
    historical = pd.DataFrame({"SINGLE": returns}, index=dates)
    predicted = pd.Series([0.05], index=["SINGLE"])
    return predicted, historical


@pytest.fixture
def perfectly_correlated_assets():
    """Generate perfectly correlated assets (singular covariance)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    base_returns = np.random.randn(100) * 0.02
    historical = pd.DataFrame(
        {
            "ASSET1": base_returns,
            "ASSET2": base_returns,  # Identical = perfectly correlated
        },
        index=dates,
    )
    predicted = pd.Series([0.05, 0.03], index=["ASSET1", "ASSET2"])
    return predicted, historical


@pytest.fixture
def high_volatility_returns(sample_dates):
    """Generate high volatility returns (stress test)."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.10  # 10% daily vol
    return pd.Series(returns, index=sample_dates)


@pytest.fixture
def trending_prices():
    """Generate strongly trending prices."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Strong upward trend
    prices = 100 * (1.002 ** np.arange(100))  # ~0.2% daily growth
    np.random.seed(42)
    noise = np.random.randn(100) * 0.5
    return pd.Series(prices + noise, index=dates)


@pytest.fixture
def mean_reverting_prices():
    """Generate mean-reverting prices."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    # Ornstein-Uhlenbeck-like process
    prices = [100.0]
    for _ in range(99):
        # Mean reversion toward 100
        reversion = 0.1 * (100 - prices[-1])
        noise = np.random.randn() * 0.5
        prices.append(prices[-1] + reversion + noise)
    return pd.Series(prices, index=dates)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "optional: marks tests that require optional dependencies",
    )


# Helper functions for tests
def assert_series_equal_approx(series1: pd.Series, series2: pd.Series, rtol: float = 1e-5):
    """Assert two series are approximately equal."""
    pd.testing.assert_index_equal(series1.index, series2.index)
    np.testing.assert_allclose(series1.values, series2.values, rtol=rtol)


def assert_valid_predictions(predictions: pd.Series, horizon: int):
    """Assert predictions are valid."""
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == horizon
    assert not predictions.isna().any()
    assert isinstance(predictions.index, pd.DatetimeIndex)


def assert_valid_weights(weights: pd.Series, tickers: list = None):
    """Assert portfolio weights are valid."""
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert all(weights >= -1e-6)  # Allow small numerical errors

    if tickers is not None:
        assert len(weights) == len(tickers)
        assert all(ticker in weights.index for ticker in tickers)
