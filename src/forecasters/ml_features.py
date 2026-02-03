"""
Feature Engineering for ML Forecasters

This module provides feature extraction functions for transforming time series
into supervised learning problems.

Feature types:
1. Lag features: r(t-1), r(t-2), ..., r(t-n)
2. Rolling statistics: mean, std, min, max over windows
3. Technical indicators: RSI, MACD, etc. (optional)
4. Time features: day_of_week, month, etc.

Usage:
    >>> features_df = create_features(price_series, lags=5, windows=[5, 20])
    >>> X, y = prepare_ml_data(features_df, horizon=1)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def create_lag_features(series: pd.Series, lags: int = 5, diff: bool = False) -> pd.DataFrame:
    """
    Create lagged features.

    Args:
        series: Time series (prices or returns)
        lags: Number of lags to create
        diff: If True, use differenced values (returns)

    Returns:
        DataFrame with lag columns
    """
    df = pd.DataFrame(index=series.index)

    if diff:
        series = series.pct_change()

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    return df


def create_rolling_features(
    series: pd.Series, windows: list[int] | None = None, diff: bool = False
) -> pd.DataFrame:
    """
    Create rolling window statistics.

    Args:
        series: Time series
        windows: List of window sizes
        diff: If True, use returns

    Returns:
        DataFrame with rolling features
    """
    if windows is None:
        windows = [5, 20, 60]
    df = pd.DataFrame(index=series.index)

    if diff:
        series = series.pct_change()

    for window in windows:
        df[f"mean_{window}"] = series.rolling(window).mean()
        df[f"std_{window}"] = series.rolling(window).std()
        df[f"min_{window}"] = series.rolling(window).min()
        df[f"max_{window}"] = series.rolling(window).max()

        # Z-score relative to window
        df[f"zscore_{window}"] = (series - df[f"mean_{window}"]) / (df[f"std_{window}"] + 1e-8)

    return df


def create_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create time-based features.

    Args:
        index: DatetimeIndex

    Returns:
        DataFrame with time features
    """
    df = pd.DataFrame(index=index)

    # Day of week (Monday=0, Sunday=6)
    df["day_of_week"] = index.dayofweek

    # Month (1-12)
    df["month"] = index.month

    # Quarter (1-4)
    df["quarter"] = index.quarter

    # Day of month
    df["day_of_month"] = index.day

    # Is month end
    df["is_month_end"] = index.is_month_end.astype(int)

    # Is month start
    df["is_month_start"] = index.is_month_start.astype(int)

    # Is quarter end
    df["is_quarter_end"] = index.is_quarter_end.astype(int)

    return df


def create_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        series: Price series
        window: RSI window (default: 14)

    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD indicator.

    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        DataFrame with MACD and signal columns
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df = pd.DataFrame(index=series.index)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_diff"] = macd_line - signal_line

    return df


def create_features(
    series: pd.Series,
    lags: int = 5,
    windows: list[int] | None = None,
    include_time: bool = True,
    include_technical: bool = False,
    diff: bool = True,
) -> pd.DataFrame:
    """
    Create comprehensive feature set for ML models.

    Args:
        series: Time series (prices)
        lags: Number of lag features
        windows: Rolling window sizes
        include_time: Include time-based features
        include_technical: Include technical indicators (RSI, MACD)
        diff: Use returns instead of prices for features

    Returns:
        DataFrame with all features
    """
    if windows is None:
        windows = [5, 20]
    features = []

    # Lag features
    lag_features = create_lag_features(series, lags=lags, diff=diff)
    features.append(lag_features)

    # Rolling features
    rolling_features = create_rolling_features(series, windows=windows, diff=diff)
    features.append(rolling_features)

    # Time features
    if include_time:
        time_features = create_time_features(series.index)
        features.append(time_features)

    # Technical indicators
    if include_technical:
        rsi = create_rsi(series)
        features.append(pd.DataFrame({"rsi": rsi}, index=series.index))

        macd = create_macd(series)
        features.append(macd)

    # Combine all features
    features_df = pd.concat(features, axis=1)

    return features_df


def prepare_ml_data(
    series: pd.Series,
    lags: int = 5,
    windows: list[int] | None = None,
    horizon: int = 1,
    include_time: bool = True,
    include_technical: bool = False,
    diff: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for ML model training.

    Args:
        series: Time series (prices)
        lags: Number of lag features
        windows: Rolling window sizes
        horizon: Forecast horizon (target = series.shift(-horizon))
        include_time: Include time features
        include_technical: Include technical indicators
        diff: Use returns

    Returns:
        Tuple of (X features, y target)
    """
    # Create features
    if windows is None:
        windows = [5, 20]
    X = create_features(
        series,
        lags=lags,
        windows=windows,
        include_time=include_time,
        include_technical=include_technical,
        diff=diff,
    )

    # Create target (future return or price)
    if diff:
        # Forward return: r_{t→t+h} = (S_{t+h} - S_t) / S_t = S_{t+h}/S_t - 1
        # Note: pct_change().shift(-h) is WRONG - it gives r_{t+h-1→t+h}, not r_{t→t+h}
        y = (series.shift(-horizon) / series) - 1
    else:
        y = series.shift(-horizon)

    # Align and drop NaN
    data = pd.concat([X, y.rename("target")], axis=1)
    data = data.dropna()

    X_clean = data.drop("target", axis=1)
    y_clean = data["target"]

    logger.debug(
        f"Prepared ML data: {len(X_clean)} samples, {X_clean.shape[1]} features, "
        f"horizon={horizon}"
    )

    return X_clean, y_clean


def prepare_combined_ml_data(
    price_dict: dict[str, pd.Series],
    lags: int = 5,
    windows: list[int] | None = None,
    horizon: int = 1,
    include_time: bool = True,
    include_technical: bool = False,
    diff: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare combined data from multiple tickers for a single ML model.

    This function pools data from all tickers to train one model instead of
    training separate models per ticker. The model learns general patterns
    that apply across all assets.

    Advantages:
    - More training data (N tickers * T time steps)
    - Model learns cross-asset patterns
    - Faster training (one model vs N models)
    - Better generalization for similar assets

    Args:
        price_dict: Dictionary of {ticker: price_series}
        lags: Number of lag features
        windows: Rolling window sizes
        horizon: Forecast horizon
        include_time: Include time features
        include_technical: Include technical indicators
        diff: Use returns (True) or prices (False)

    Returns:
        Tuple of (X features, y target, ticker_labels)
        ticker_labels allows you to split predictions back by ticker
    """
    if windows is None:
        windows = [5, 20]

    all_X = []
    all_y = []
    all_tickers = []

    for ticker, series in price_dict.items():
        try:
            X, y = prepare_ml_data(
                series,
                lags=lags,
                windows=windows,
                horizon=horizon,
                include_time=include_time,
                include_technical=include_technical,
                diff=diff,
            )

            all_X.append(X)
            all_y.append(y)
            all_tickers.extend([ticker] * len(X))

        except Exception as e:
            logger.warning(f"Failed to prepare data for {ticker}: {e}")
            continue

    if not all_X:
        raise ValueError("No valid data prepared from any ticker")

    # Combine all data
    X_combined = pd.concat(all_X, axis=0, ignore_index=True)
    y_combined = pd.concat(all_y, axis=0, ignore_index=True)
    ticker_labels = pd.Series(all_tickers, index=X_combined.index)

    logger.info(
        f"Combined data from {len(price_dict)} tickers: "
        f"{len(X_combined)} samples, {X_combined.shape[1]} features"
    )

    return X_combined, y_combined, ticker_labels


__all__ = [
    "create_lag_features",
    "create_rolling_features",
    "create_time_features",
    "create_rsi",
    "create_macd",
    "create_features",
    "prepare_ml_data",
    "prepare_combined_ml_data",
]
