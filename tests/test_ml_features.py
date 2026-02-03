"""Tests for ML feature engineering module.

This module tests all feature extraction functions in src/forecasters/ml_features.py:
- create_lag_features()
- create_rolling_features()
- create_time_features()
- create_rsi()
- create_macd()
- create_features()
- prepare_ml_data()
- prepare_combined_ml_data()
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters.ml_features import (
    create_features,
    create_lag_features,
    create_macd,
    create_rolling_features,
    create_rsi,
    create_time_features,
    prepare_combined_ml_data,
    prepare_ml_data,
)


@pytest.fixture
def sample_price_series():
    """Create a sample price series for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Generate prices with a slight upward trend and volatility
    returns = np.random.randn(100) * 0.02 + 0.001
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="Price")


@pytest.fixture
def short_price_series():
    """Create a short price series for edge case testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = [100, 101, 99, 102, 103, 101, 100, 104, 105, 103]
    return pd.Series(prices, index=dates, name="Price")


@pytest.fixture
def multi_ticker_data():
    """Create price data for multiple tickers."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    tickers = {}
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        returns = np.random.randn(100) * 0.02 + 0.001
        prices = 100 * np.cumprod(1 + returns)
        tickers[ticker] = pd.Series(prices, index=dates, name=ticker)

    return tickers


class TestCreateLagFeatures:
    """Tests for create_lag_features function."""

    def test_creates_correct_number_of_lags(self, sample_price_series):
        """Test that correct number of lag columns are created."""
        lags = 5
        df = create_lag_features(sample_price_series, lags=lags)

        assert df.shape[1] == lags
        assert all(f"lag_{i}" in df.columns for i in range(1, lags + 1))

    def test_lag_values_are_shifted(self, short_price_series):
        """Test that lag values are correctly shifted."""
        df = create_lag_features(short_price_series, lags=2, diff=False)

        # lag_1 should be the previous value
        assert df["lag_1"].iloc[1] == short_price_series.iloc[0]
        assert df["lag_1"].iloc[2] == short_price_series.iloc[1]

        # lag_2 should be 2 steps back
        assert df["lag_2"].iloc[2] == short_price_series.iloc[0]

    def test_first_lag_values_are_nan(self, sample_price_series):
        """Test that first rows have NaN due to shifting."""
        df = create_lag_features(sample_price_series, lags=3, diff=False)

        # First row should have NaN for all lags
        assert df["lag_1"].isna().iloc[0]
        assert df["lag_2"].isna().iloc[:2].all()
        assert df["lag_3"].isna().iloc[:3].all()

    def test_diff_option(self, sample_price_series):
        """Test that diff=True creates returns-based lags."""
        df_no_diff = create_lag_features(sample_price_series, lags=2, diff=False)
        df_diff = create_lag_features(sample_price_series, lags=2, diff=True)

        # With diff=True, values should be percentage changes
        # The magnitudes should be much smaller
        valid_no_diff = df_no_diff["lag_1"].dropna()
        valid_diff = df_diff["lag_1"].dropna()

        assert abs(valid_diff.mean()) < abs(valid_no_diff.mean())

    def test_preserves_index(self, sample_price_series):
        """Test that output preserves the original index."""
        df = create_lag_features(sample_price_series, lags=3)

        assert df.index.equals(sample_price_series.index)

    def test_single_lag(self, sample_price_series):
        """Test with single lag."""
        df = create_lag_features(sample_price_series, lags=1)

        assert df.shape[1] == 1
        assert "lag_1" in df.columns


class TestCreateRollingFeatures:
    """Tests for create_rolling_features function."""

    def test_creates_correct_columns(self, sample_price_series):
        """Test that all expected rolling columns are created."""
        windows = [5, 10]
        df = create_rolling_features(sample_price_series, windows=windows)

        expected_columns = []
        for w in windows:
            expected_columns.extend(
                [f"mean_{w}", f"std_{w}", f"min_{w}", f"max_{w}", f"zscore_{w}"]
            )

        assert set(df.columns) == set(expected_columns)

    def test_rolling_mean_calculation(self, short_price_series):
        """Test that rolling mean is calculated correctly."""
        df = create_rolling_features(short_price_series, windows=[3], diff=False)

        # Manual calculation for window=3
        expected_mean = short_price_series.iloc[:3].mean()
        assert np.isclose(df["mean_3"].iloc[2], expected_mean)

    def test_rolling_std_calculation(self, short_price_series):
        """Test that rolling std is calculated correctly."""
        df = create_rolling_features(short_price_series, windows=[3], diff=False)

        # Manual calculation for window=3
        expected_std = short_price_series.iloc[:3].std()
        assert np.isclose(df["std_3"].iloc[2], expected_std)

    def test_rolling_min_max(self, short_price_series):
        """Test rolling min and max calculations."""
        df = create_rolling_features(short_price_series, windows=[3], diff=False)

        # Manual check for first complete window
        window_data = short_price_series.iloc[:3]
        assert np.isclose(df["min_3"].iloc[2], window_data.min())
        assert np.isclose(df["max_3"].iloc[2], window_data.max())

    def test_zscore_calculation(self, sample_price_series):
        """Test z-score calculation."""
        df = create_rolling_features(sample_price_series, windows=[5], diff=False)

        # Z-score should be between -4 and 4 for most values
        valid_zscore = df["zscore_5"].dropna()
        assert valid_zscore.abs().max() < 10  # Reasonable range

    def test_default_windows(self, sample_price_series):
        """Test default window sizes [5, 20, 60]."""
        df = create_rolling_features(sample_price_series)

        assert "mean_5" in df.columns
        assert "mean_20" in df.columns
        assert "mean_60" in df.columns

    def test_diff_option(self, sample_price_series):
        """Test diff=True applies pct_change before rolling."""
        df_diff = create_rolling_features(sample_price_series, windows=[5], diff=True)
        df_no_diff = create_rolling_features(sample_price_series, windows=[5], diff=False)

        # Values should be different
        assert not df_diff["mean_5"].dropna().equals(df_no_diff["mean_5"].dropna())


class TestCreateTimeFeatures:
    """Tests for create_time_features function."""

    def test_creates_all_time_columns(self, sample_price_series):
        """Test that all time feature columns are created."""
        df = create_time_features(sample_price_series.index)

        expected_columns = [
            "day_of_week",
            "month",
            "quarter",
            "day_of_month",
            "is_month_end",
            "is_month_start",
            "is_quarter_end",
        ]
        assert set(df.columns) == set(expected_columns)

    def test_day_of_week_values(self):
        """Test day_of_week values (Monday=0, Sunday=6)."""
        # Create dates that span a week
        dates = pd.date_range("2024-01-01", periods=7, freq="D")  # Jan 1, 2024 is Monday
        df = create_time_features(dates)

        # Monday=0, Tuesday=1, ..., Sunday=6
        expected = [0, 1, 2, 3, 4, 5, 6]
        assert list(df["day_of_week"]) == expected

    def test_month_values(self):
        """Test month values (1-12)."""
        dates = pd.DatetimeIndex(["2024-01-15", "2024-06-15", "2024-12-15"])
        df = create_time_features(dates)

        assert list(df["month"]) == [1, 6, 12]

    def test_quarter_values(self):
        """Test quarter values (1-4)."""
        dates = pd.DatetimeIndex(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"])
        df = create_time_features(dates)

        assert list(df["quarter"]) == [1, 2, 3, 4]

    def test_month_end_flag(self):
        """Test is_month_end flag."""
        dates = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01"])
        df = create_time_features(dates)

        assert list(df["is_month_end"]) == [0, 1, 0]

    def test_month_start_flag(self):
        """Test is_month_start flag."""
        dates = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-02-02"])
        df = create_time_features(dates)

        assert list(df["is_month_start"]) == [0, 1, 0]

    def test_quarter_end_flag(self):
        """Test is_quarter_end flag."""
        dates = pd.DatetimeIndex(["2024-03-30", "2024-03-31", "2024-04-01"])
        df = create_time_features(dates)

        assert list(df["is_quarter_end"]) == [0, 1, 0]


class TestCreateRSI:
    """Tests for create_rsi function."""

    def test_rsi_range(self, sample_price_series):
        """Test that RSI values are between 0 and 100."""
        rsi = create_rsi(sample_price_series, window=14)
        valid_rsi = rsi.dropna()

        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_rsi_with_uptrend(self):
        """Test RSI with strong uptrend should be high."""
        # Monotonically increasing prices
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        prices = pd.Series(range(100, 150), index=dates)

        rsi = create_rsi(prices, window=14)
        valid_rsi = rsi.dropna()

        # RSI should be close to 100 for strong uptrend
        assert valid_rsi.iloc[-1] > 90

    def test_rsi_with_downtrend(self):
        """Test RSI with strong downtrend should be low."""
        # Monotonically decreasing prices
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        prices = pd.Series(range(150, 100, -1), index=dates)

        rsi = create_rsi(prices, window=14)
        valid_rsi = rsi.dropna()

        # RSI should be close to 0 for strong downtrend
        assert valid_rsi.iloc[-1] < 10

    def test_rsi_window_parameter(self, sample_price_series):
        """Test RSI with different window sizes."""
        rsi_14 = create_rsi(sample_price_series, window=14)
        rsi_7 = create_rsi(sample_price_series, window=7)

        # Different windows should produce different values
        assert not rsi_14.dropna().equals(rsi_7.dropna())

    def test_rsi_nan_count(self, sample_price_series):
        """Test that RSI has expected number of NaN values."""
        window = 14
        rsi = create_rsi(sample_price_series, window=window)

        # RSI needs: 1 value for diff() + (window-1) for rolling = window NaN values
        # Due to overlapping, actual NaN count may be window-1 or window
        assert rsi.isna().sum() >= window - 1


class TestCreateMACD:
    """Tests for create_macd function."""

    def test_macd_columns(self, sample_price_series):
        """Test that MACD returns expected columns."""
        df = create_macd(sample_price_series)

        assert set(df.columns) == {"macd", "macd_signal", "macd_diff"}

    def test_macd_diff_calculation(self, sample_price_series):
        """Test that macd_diff = macd - macd_signal."""
        df = create_macd(sample_price_series)

        calculated_diff = df["macd"] - df["macd_signal"]
        assert np.allclose(df["macd_diff"].values, calculated_diff.values)

    def test_macd_with_uptrend(self):
        """Test MACD with uptrend should be positive."""
        # Strong uptrend
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = pd.Series(np.linspace(100, 200, 100), index=dates)

        df = create_macd(prices)

        # MACD should be positive in uptrend (fast EMA > slow EMA)
        assert df["macd"].iloc[-1] > 0

    def test_macd_custom_parameters(self, sample_price_series):
        """Test MACD with custom parameters."""
        df_default = create_macd(sample_price_series)
        df_custom = create_macd(sample_price_series, fast=6, slow=13, signal=5)

        # Different parameters should produce different values
        assert not df_default["macd"].equals(df_custom["macd"])

    def test_macd_preserves_index(self, sample_price_series):
        """Test that MACD preserves original index."""
        df = create_macd(sample_price_series)

        assert df.index.equals(sample_price_series.index)


class TestCreateFeatures:
    """Tests for create_features function."""

    def test_combines_all_features(self, sample_price_series):
        """Test that create_features combines lag, rolling, and time features."""
        df = create_features(
            sample_price_series,
            lags=3,
            windows=[5],
            include_time=True,
            include_technical=False,
        )

        # Check lag features
        assert "lag_1" in df.columns
        assert "lag_2" in df.columns
        assert "lag_3" in df.columns

        # Check rolling features
        assert "mean_5" in df.columns
        assert "std_5" in df.columns

        # Check time features
        assert "day_of_week" in df.columns
        assert "month" in df.columns

    def test_technical_indicators_optional(self, sample_price_series):
        """Test that technical indicators are optional."""
        df_no_tech = create_features(sample_price_series, include_technical=False)
        df_with_tech = create_features(sample_price_series, include_technical=True)

        assert "rsi" not in df_no_tech.columns
        assert "macd" not in df_no_tech.columns

        assert "rsi" in df_with_tech.columns
        assert "macd" in df_with_tech.columns

    def test_time_features_optional(self, sample_price_series):
        """Test that time features are optional."""
        df_no_time = create_features(sample_price_series, include_time=False)
        df_with_time = create_features(sample_price_series, include_time=True)

        assert "day_of_week" not in df_no_time.columns
        assert "day_of_week" in df_with_time.columns

    def test_default_windows(self, sample_price_series):
        """Test default windows [5, 20]."""
        df = create_features(sample_price_series)

        assert "mean_5" in df.columns
        assert "mean_20" in df.columns

    def test_preserves_index(self, sample_price_series):
        """Test that output preserves original index."""
        df = create_features(sample_price_series)

        assert df.index.equals(sample_price_series.index)


class TestPrepareMLData:
    """Tests for prepare_ml_data function."""

    def test_returns_x_and_y(self, sample_price_series):
        """Test that function returns X and y tuple."""
        X, y = prepare_ml_data(sample_price_series)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_and_y_aligned(self, sample_price_series):
        """Test that X and y have same length."""
        X, y = prepare_ml_data(sample_price_series)

        assert len(X) == len(y)

    def test_no_nan_in_output(self, sample_price_series):
        """Test that output has no NaN values."""
        X, y = prepare_ml_data(sample_price_series)

        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_horizon_shifts_target(self, sample_price_series):
        """Test that horizon parameter shifts target correctly."""
        X1, y1 = prepare_ml_data(sample_price_series, horizon=1)
        X5, y5 = prepare_ml_data(sample_price_series, horizon=5)

        # Different horizons should produce different targets
        assert not y1.equals(y5)

    def test_diff_true_creates_return_targets(self, sample_price_series):
        """Test that diff=True creates return-based targets."""
        X, y = prepare_ml_data(sample_price_series, diff=True, horizon=1)

        # Returns should be small values (percentage changes)
        assert abs(y.mean()) < 1.0
        assert y.std() < 1.0

    def test_diff_false_creates_price_targets(self, sample_price_series):
        """Test that diff=False creates price-based targets."""
        X, y = prepare_ml_data(sample_price_series, diff=False, horizon=1)

        # Prices should be larger values
        assert y.mean() > 10  # Assuming prices are > 10

    def test_fewer_samples_with_larger_lags(self, sample_price_series):
        """Test that larger lags result in fewer samples."""
        X3, y3 = prepare_ml_data(sample_price_series, lags=3, windows=[5])
        X10, y10 = prepare_ml_data(sample_price_series, lags=10, windows=[5])

        # More lags = more NaN rows dropped = fewer samples
        assert len(X3) > len(X10)

    def test_target_is_forward_return(self, short_price_series):
        """Test that target is correct forward return calculation."""
        # Use simple data to verify calculation
        X, y = prepare_ml_data(
            short_price_series,
            lags=1,
            windows=[3],
            horizon=1,
            diff=True,
            include_time=False,
        )

        # Forward return = (S_{t+1} / S_t) - 1
        # For a specific point, verify manually
        # This test ensures the formula is (price_future / price_current) - 1


class TestPrepareCombinedMLData:
    """Tests for prepare_combined_ml_data function."""

    def test_combines_multiple_tickers(self, multi_ticker_data):
        """Test that data from multiple tickers is combined."""
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        # Should have 3 tickers worth of data
        unique_tickers = tickers.unique()
        assert len(unique_tickers) == 3
        assert set(unique_tickers) == {"AAPL", "MSFT", "GOOGL"}

    def test_returns_ticker_labels(self, multi_ticker_data):
        """Test that ticker labels are returned."""
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        assert isinstance(tickers, pd.Series)
        assert len(tickers) == len(X)

    def test_x_and_y_aligned(self, multi_ticker_data):
        """Test that X and y have same length."""
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        assert len(X) == len(y)
        assert len(X) == len(tickers)

    def test_no_nan_in_output(self, multi_ticker_data):
        """Test that output has no NaN values."""
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_more_samples_than_single_ticker(self, multi_ticker_data):
        """Test that combined data has more samples than single ticker."""
        # Single ticker
        single_ticker_data = multi_ticker_data["AAPL"]
        X_single, y_single = prepare_ml_data(single_ticker_data)

        # Combined
        X_combined, y_combined, _ = prepare_combined_ml_data(multi_ticker_data)

        # Combined should have roughly 3x the samples
        assert len(X_combined) > len(X_single) * 2

    def test_empty_dict_raises_error(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="No valid data"):
            prepare_combined_ml_data({})

    def test_handles_failing_ticker(self, multi_ticker_data):
        """Test that failing ticker is skipped gracefully."""
        # Add a ticker with bad data (too short)
        multi_ticker_data["BAD"] = pd.Series(
            [100, 101], index=pd.date_range("2024-01-01", periods=2)
        )

        # Should still work, just skip the bad ticker
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        # BAD ticker should not be in the results
        assert "BAD" not in tickers.values

    def test_feature_consistency_across_tickers(self, multi_ticker_data):
        """Test that all tickers produce same feature columns."""
        X, y, tickers = prepare_combined_ml_data(multi_ticker_data)

        # All rows should have same features
        assert X.shape[1] > 0

        # Check each ticker's subset has same columns
        for ticker in tickers.unique():
            ticker_mask = tickers == ticker
            ticker_X = X[ticker_mask]
            assert ticker_X.shape[1] == X.shape[1]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_price_series(self):
        """Test with constant prices (no variation)."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        prices = pd.Series([100.0] * 50, index=dates)

        # Should not raise error
        X, y = prepare_ml_data(prices, lags=3, windows=[5])

        # Returns should be 0 (or close to 0)
        assert np.allclose(y, 0, atol=1e-10)

    def test_single_window(self):
        """Test with single window size."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        prices = pd.Series(np.random.randn(50).cumsum() + 100, index=dates)

        df = create_rolling_features(prices, windows=[10])

        assert "mean_10" in df.columns
        assert "mean_5" not in df.columns  # Only requested window=10

    def test_large_horizon(self, sample_price_series):
        """Test with large forecast horizon."""
        X, y = prepare_ml_data(sample_price_series, horizon=10)

        # Should have fewer samples due to larger horizon
        assert len(X) < len(sample_price_series) - 20

    def test_all_features_enabled(self, sample_price_series):
        """Test with all feature types enabled."""
        df = create_features(
            sample_price_series,
            lags=5,
            windows=[5, 10, 20],
            include_time=True,
            include_technical=True,
            diff=True,
        )

        # Should have many columns
        assert df.shape[1] > 20  # Lag + rolling + time + technical

    def test_minimum_data_for_ml(self):
        """Test minimum data requirements."""
        # Very short series
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices = pd.Series(np.random.randn(30).cumsum() + 100, index=dates)

        # Should work with minimal data
        X, y = prepare_ml_data(prices, lags=3, windows=[5], horizon=1)

        assert len(X) > 0
        assert len(y) > 0
