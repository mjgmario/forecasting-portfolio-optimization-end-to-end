"""
Forecasting Evaluation Metrics

Comprehensive metrics for evaluating forecast quality.

Metrics implemented:
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Squared Error)
3. MASE (Mean Absolute Scaled Error) - scales against naive forecast
4. Directional Accuracy - % of correctly predicted directions
5. Hit Rate - same as directional accuracy
6. Pinball Loss - for quantile forecasts
7. Tracking Error - for portfolio returns

Why these metrics:
- MAE/RMSE: Standard error metrics
- MASE: Benchmark against naive (critical for time series)
- Directional Accuracy: Most important for trading/portfolio
- Pinball: For risk-aware forecasting
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_numpy(arr):
    """Convert pandas Series or numpy array to numpy array."""
    if hasattr(arr, "values"):
        return arr.values
    return np.asarray(arr)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|y_true - y_pred|)

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: MAE value (lower is better)
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((y_true - y_pred)²))

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: RMSE value (lower is better)
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: MAPE value in % (lower is better)
    :rtype: float

    .. note:: Undefined when y_true contains zeros
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning("MAPE undefined: y_true contains only zeros")
        return np.inf

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(
    y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series | None = None, seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE scales the forecast error against a naive forecast baseline.

    MASE = MAE(forecast) / MAE(naive)

    Where naive forecast is:
    - For non-seasonal: y(t) = y(t-1)
    - For seasonal: y(t) = y(t-seasonality)

    Interpretation:
    - MASE < 1: Forecast beats naive
    - MASE = 1: Forecast equals naive
    - MASE > 1: Forecast worse than naive

    :param y_true: Actual values (test set)
    :type y_true: pd.Series
    :param y_pred: Predicted values (test set)
    :type y_pred: pd.Series
    :param y_train: Training data for computing naive baseline. If None, uses y_true for in-sample MASE
    :type y_train: pd.Series, optional
    :param seasonality: Seasonal period (default: 1 = non-seasonal)
    :type seasonality: int
    :return: MASE value (lower is better, < 1 is good)
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # MAE of forecast
    forecast_mae = mae(y_true, y_pred)

    # MAE of naive baseline
    # Use y_train if provided, otherwise y_true for in-sample MASE
    arr = _to_numpy(y_train if y_train is not None else y_true)

    # Check sufficient data for seasonal naive
    if len(arr) <= seasonality:
        logger.warning(f"MASE undefined: insufficient data for seasonality={seasonality}")
        return np.nan

    # Correct seasonal lag calculation: y_t - y_{t-m}
    # Note: np.diff(arr, n=m) computes m-th order difference, NOT lag m
    naive_errors = np.abs(arr[seasonality:] - arr[:-seasonality])
    naive_mae = float(np.mean(naive_errors))

    if naive_mae == 0:
        logger.warning("MASE undefined: naive forecast has zero error")
        return np.inf

    return float(forecast_mae / naive_mae)


def directional_accuracy_1step(
    y_prev: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Directional accuracy for horizon=1 walk-forward backtesting.

    Compares if the predicted movement has the same sign as the actual movement,
    relative to the previous value (last value from training data).

    - true_move = y_true_t - y_prev_t
    - pred_move = y_pred_t - y_prev_t
    - Hit if sign(true_move) == sign(pred_move)

    This metric is useful for test_size=1 backtesting where the standard
    directional_accuracy (which needs 2+ observations) cannot be computed.

    :param y_prev: Last value from train at each fold (aligned with y_true)
    :type y_prev: pd.Series or np.ndarray
    :param y_true: Actual test values
    :type y_true: pd.Series or np.ndarray
    :param y_pred: Predicted values
    :type y_pred: pd.Series or np.ndarray
    :return: Fraction of correct directional predictions (0-1)
    :rtype: float
    """
    y_prev_arr = _to_numpy(y_prev)
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    if len(y_prev_arr) != len(y_true_arr) or len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_prev, y_true, and y_pred must have the same length")

    if len(y_true_arr) == 0:
        return np.nan

    true_move = y_true_arr - y_prev_arr
    pred_move = y_pred_arr - y_prev_arr

    return float(np.mean(np.sign(true_move) == np.sign(pred_move)))


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Directional Accuracy (Hit Rate).

    Fraction of times the forecast correctly predicts the direction
    of change (up/down).

    DA = mean(sign(Δy_true) == sign(Δy_pred))

    This is THE most important metric for trading/portfolio:
    - 0.5 = random (no skill)
    - >0.55 = good for financial markets
    - >0.60 = excellent

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: Directional accuracy as fraction (0-1)
    :rtype: float
    :raises ValueError: If fewer than 2 observations

    .. note:: Requires at least 2 observations
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        raise ValueError("Need at least 2 observations for directional accuracy")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Compute changes
    true_direction = np.sign(np.diff(_to_numpy(y_true)))
    pred_direction = np.sign(np.diff(_to_numpy(y_pred)))

    # Count correct predictions
    correct = (true_direction == pred_direction).sum()
    total = len(true_direction)

    return float(correct / total)


def hit_rate(
    returns_true: pd.Series, returns_pred: pd.Series, threshold: float | None = None
) -> float:
    """
    Hit Rate for returns.

    If threshold is None: Fraction of times forecast correctly predicts sign (positive/negative).
    If threshold is provided: Fraction of times forecast is within threshold of actual value.

    HR = mean(sign(r_true) == sign(r_pred))  [if threshold=None]
    HR = mean(|r_true - r_pred| / |r_true| <= threshold)  [if threshold provided]

    :param returns_true: Actual returns
    :type returns_true: pd.Series
    :param returns_pred: Predicted returns
    :type returns_pred: pd.Series
    :param threshold: Optional threshold for relative error (e.g., 0.01 for 1%)
    :type threshold: float, optional
    :return: Hit rate as fraction (0-1)
    :rtype: float
    """
    if len(returns_true) != len(returns_pred):
        raise ValueError("returns_true and returns_pred must have same length")

    true_arr = _to_numpy(returns_true)
    pred_arr = _to_numpy(returns_pred)

    if threshold is not None:
        # Check if predictions are within threshold
        relative_errors = np.abs(true_arr - pred_arr) / np.abs(true_arr)
        correct = (relative_errors <= threshold).sum()
    else:
        # Check if signs match
        correct = (np.sign(true_arr) == np.sign(pred_arr)).sum()

    total = len(returns_true)
    return float(correct / total)


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float = 0.5) -> float:
    """
    Pinball Loss for quantile forecasts.

    Asymmetric loss function that penalizes under/over-prediction differently.

    L_q(y, ŷ) = (1-q) * (y - ŷ) if y >= ŷ  (underestimation)
                 q * (ŷ - y) if y < ŷ       (overestimation)

    For median (q=0.5), this is equivalent to MAE.

    For low quantiles (e.g., q=0.1):
    - Underestimation is penalized more (factor: 1-q = 0.9)
    - Overestimation is penalized less (factor: q = 0.1)

    For high quantiles (e.g., q=0.9):
    - Underestimation is penalized less (factor: 1-q = 0.1)
    - Overestimation is penalized more (factor: q = 0.9)

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted quantile values
    :type y_pred: pd.Series
    :param quantile: Quantile level (default: 0.5 = median)
    :type quantile: float
    :return: Pinball loss (lower is better)
    :rtype: float
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be between 0 and 1")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    errors = _to_numpy(y_true) - _to_numpy(y_pred)
    # When errors >= 0 (y_true > y_pred, underestimation): (1-q) * error
    # When errors < 0 (y_true < y_pred, overestimation): q * |error|
    loss = np.where(errors >= 0, (1 - quantile) * errors, quantile * (-errors))

    return float(np.mean(loss))


def tracking_error(
    returns_portfolio: pd.Series, returns_benchmark: pd.Series, annualize: bool = True
) -> float:
    """
    Tracking Error (Ex-Post).

    Standard deviation of the difference between portfolio and benchmark returns.

    TE = std(R_portfolio - R_benchmark)

    Measures how closely portfolio tracks benchmark.

    :param returns_portfolio: Portfolio returns
    :type returns_portfolio: pd.Series
    :param returns_benchmark: Benchmark returns
    :type returns_benchmark: pd.Series
    :param annualize: If True, annualize the tracking error (default: True)
    :type annualize: bool
    :return: Tracking error (annualized if specified)
    :rtype: float
    """
    if len(returns_portfolio) != len(returns_benchmark):
        raise ValueError("Portfolio and benchmark returns must have same length")

    active_returns = returns_portfolio - returns_benchmark
    te = float(active_returns.std())

    if annualize:
        # Assume daily returns
        te *= np.sqrt(252)

    return te


def information_ratio(
    returns_portfolio: pd.Series, returns_benchmark: pd.Series, annualize: bool = True
) -> float:
    """
    Information Ratio.

    IR = E[R_portfolio - R_benchmark] / TrackingError

    Measures risk-adjusted active returns.

    :param returns_portfolio: Portfolio returns
    :type returns_portfolio: pd.Series
    :param returns_benchmark: Benchmark returns
    :type returns_benchmark: pd.Series
    :param annualize: If True, annualize the ratio (default: True)
    :type annualize: bool
    :return: Information ratio (higher is better)
    :rtype: float
    """
    active_returns = returns_portfolio - returns_benchmark
    mean_active = float(active_returns.mean())
    std_active = float(active_returns.std())

    if std_active == 0:
        return np.inf if mean_active > 0 else -np.inf

    ir = mean_active / std_active

    if annualize:
        ir *= np.sqrt(252)

    return ir


# =============================================================================
# PANEL / CROSS-SECTIONAL METRICS (for multi-asset evaluation)
# =============================================================================


def ic_pearson(pred_returns: pd.DataFrame, real_returns: pd.DataFrame) -> pd.Series:
    """
    Information Coefficient (Pearson) - cross-sectional correlation per date.

    Measures how well predicted returns rank assets compared to actual returns.
    Computed as Pearson correlation between predicted and realized returns
    across all assets for each date.

    :param pred_returns: DataFrame (Date × Ticker) of predicted returns
    :type pred_returns: pd.DataFrame
    :param real_returns: DataFrame (Date × Ticker) of realized returns
    :type real_returns: pd.DataFrame
    :return: Series of IC values per date
    :rtype: pd.Series
    """
    if pred_returns.shape != real_returns.shape:
        raise ValueError("pred_returns and real_returns must have the same shape")

    # Align indices
    common_idx = pred_returns.index.intersection(real_returns.index)
    common_cols = pred_returns.columns.intersection(real_returns.columns)

    pred = pred_returns.loc[common_idx, common_cols]
    real = real_returns.loc[common_idx, common_cols]

    ic_values = []
    for date in common_idx:
        pred_row = pred.loc[date].dropna()
        real_row = real.loc[date].dropna()
        common = pred_row.index.intersection(real_row.index)
        if len(common) >= 2:
            corr = np.corrcoef(pred_row[common], real_row[common])[0, 1]
            ic_values.append(corr)
        else:
            ic_values.append(np.nan)

    return pd.Series(ic_values, index=common_idx, name="ic_pearson")


def ic_spearman(pred_returns: pd.DataFrame, real_returns: pd.DataFrame) -> pd.Series:
    """
    Rank Information Coefficient (Spearman) - more robust to outliers.

    Measures rank correlation between predicted and realized returns
    across all assets for each date. More robust than Pearson IC.

    :param pred_returns: DataFrame (Date × Ticker) of predicted returns
    :type pred_returns: pd.DataFrame
    :param real_returns: DataFrame (Date × Ticker) of realized returns
    :type real_returns: pd.DataFrame
    :return: Series of Rank IC values per date
    :rtype: pd.Series
    """
    from scipy.stats import spearmanr

    if pred_returns.shape != real_returns.shape:
        raise ValueError("pred_returns and real_returns must have the same shape")

    common_idx = pred_returns.index.intersection(real_returns.index)
    common_cols = pred_returns.columns.intersection(real_returns.columns)

    pred = pred_returns.loc[common_idx, common_cols]
    real = real_returns.loc[common_idx, common_cols]

    ic_values = []
    for date in common_idx:
        pred_row = pred.loc[date].dropna()
        real_row = real.loc[date].dropna()
        common = pred_row.index.intersection(real_row.index)
        if len(common) >= 2:
            corr, _ = spearmanr(pred_row[common], real_row[common])
            ic_values.append(corr)
        else:
            ic_values.append(np.nan)

    return pd.Series(ic_values, index=common_idx, name="ic_spearman")


def topk_hit_rate(pred_returns: pd.DataFrame, real_returns: pd.DataFrame, k: int = 5) -> float:
    """
    Top-K Hit Rate - fraction of times predicted top-K are in actual top-K.

    Useful for evaluating stock selection strategies.

    :param pred_returns: DataFrame (Date × Ticker) of predicted returns
    :type pred_returns: pd.DataFrame
    :param real_returns: DataFrame (Date × Ticker) of realized returns
    :type real_returns: pd.DataFrame
    :param k: Number of top assets to consider (default: 5)
    :type k: int
    :return: Hit rate as fraction (0-1)
    :rtype: float
    """
    common_idx = pred_returns.index.intersection(real_returns.index)
    common_cols = pred_returns.columns.intersection(real_returns.columns)

    if len(common_cols) < k:
        logger.warning(f"topk_hit_rate: only {len(common_cols)} assets, k={k} too large")
        k = len(common_cols)

    pred = pred_returns.loc[common_idx, common_cols]
    real = real_returns.loc[common_idx, common_cols]

    hits = 0
    total = 0

    for date in common_idx:
        pred_row = pred.loc[date].dropna()
        real_row = real.loc[date].dropna()
        common = pred_row.index.intersection(real_row.index)

        if len(common) >= k:
            pred_topk = set(pred_row[common].nlargest(k).index)
            real_topk = set(real_row[common].nlargest(k).index)
            hits += len(pred_topk.intersection(real_topk))
            total += k

    return hits / total if total > 0 else np.nan


def long_short_spread(
    pred_returns: pd.DataFrame,
    real_returns: pd.DataFrame,
    quantile: float = 0.2,
) -> pd.Series:
    """
    Long-Short Spread - returns from going long top quantile, short bottom quantile.

    Measures the profitability of a simple quantile-based strategy.

    :param pred_returns: DataFrame (Date × Ticker) of predicted returns
    :type pred_returns: pd.DataFrame
    :param real_returns: DataFrame (Date × Ticker) of realized returns
    :type real_returns: pd.DataFrame
    :param quantile: Fraction for top/bottom selection (default: 0.2 = top/bottom 20%)
    :type quantile: float
    :return: Series of long-short returns per date
    :rtype: pd.Series
    """
    common_idx = pred_returns.index.intersection(real_returns.index)
    common_cols = pred_returns.columns.intersection(real_returns.columns)

    pred = pred_returns.loc[common_idx, common_cols]
    real = real_returns.loc[common_idx, common_cols]

    spreads = []
    for date in common_idx:
        pred_row = pred.loc[date].dropna()
        real_row = real.loc[date].dropna()
        common = pred_row.index.intersection(real_row.index)

        if len(common) >= 5:  # Need at least 5 assets
            n_select = max(1, int(len(common) * quantile))
            top_assets = pred_row[common].nlargest(n_select).index
            bottom_assets = pred_row[common].nsmallest(n_select).index

            long_return = real_row[top_assets].mean()
            short_return = real_row[bottom_assets].mean()
            spreads.append(long_return - short_return)
        else:
            spreads.append(np.nan)

    return pd.Series(spreads, index=common_idx, name="long_short_spread")


# =============================================================================
# ROBUST METRICS (resistant to outliers)
# =============================================================================


def median_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Median Absolute Error - more robust to outliers than MAE.

    MedAE = median(|y_true - y_pred|)

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: Median absolute error (lower is better)
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    return float(np.median(np.abs(_to_numpy(y_true) - _to_numpy(y_pred))))


def percentile_error(y_true: pd.Series, y_pred: pd.Series, percentile: float = 95) -> float:
    """
    Percentile Error - error at given percentile (useful for tail risk).

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :param percentile: Percentile to compute (default: 95 for worst 5%)
    :type percentile: float
    :return: Error at given percentile
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    errors = np.abs(_to_numpy(y_true) - _to_numpy(y_pred))
    return float(np.percentile(errors, percentile))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Symmetric Mean Absolute Percentage Error - better than MAPE for returns.

    sMAPE = mean(2 * |y - ŷ| / (|y| + |ŷ|)) * 100

    Advantages over MAPE:
    - Symmetric (penalizes over/under-prediction equally)
    - Bounded (0-200%)
    - Handles zero values better

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :return: sMAPE value in % (0-200, lower is better)
    :rtype: float
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    denominator = np.abs(y_t) + np.abs(y_p)
    # Avoid division by zero
    mask = denominator > 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(2 * np.abs(y_t[mask] - y_p[mask]) / denominator[mask]) * 100)


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series | None = None,
    metrics: list | None = None,
) -> dict:
    """
    Compute multiple forecast evaluation metrics.

    :param y_true: Actual values
    :type y_true: pd.Series
    :param y_pred: Predicted values
    :type y_pred: pd.Series
    :param y_train: Training data (for MASE)
    :type y_train: pd.Series, optional
    :param metrics: List of metric names to compute (default: all)
    :type metrics: list, optional
    :return: Dictionary of metric names to values
    :rtype: dict

    Example:
        >>> results = evaluate_forecast(y_test, predictions, y_train)
        >>> print(f"MASE: {results['mase']:.3f}")
        >>> print(f"Directional Accuracy: {results['directional_accuracy']:.1f}%")
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "mase", "directional_accuracy"]

    results = {}

    if "mae" in metrics:
        results["mae"] = mae(y_true, y_pred)

    if "rmse" in metrics:
        results["rmse"] = rmse(y_true, y_pred)

    if "mape" in metrics:
        try:
            results["mape"] = mape(y_true, y_pred)
        except Exception as e:
            logger.warning(f"MAPE calculation failed: {e}")
            results["mape"] = np.nan

    if "mase" in metrics:
        try:
            results["mase"] = mase(y_true, y_pred, y_train)
        except Exception as e:
            logger.warning(f"MASE calculation failed: {e}")
            results["mase"] = np.nan

    if "directional_accuracy" in metrics:
        try:
            results["directional_accuracy"] = directional_accuracy(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Directional accuracy calculation failed: {e}")
            results["directional_accuracy"] = np.nan

    return results


__all__ = [
    "mae",
    "rmse",
    "mape",
    "mase",
    "directional_accuracy",
    "directional_accuracy_1step",
    "hit_rate",
    "pinball_loss",
    "tracking_error",
    "information_ratio",
    "evaluate_forecast",
    # Panel / cross-sectional metrics
    "ic_pearson",
    "ic_spearman",
    "topk_hit_rate",
    "long_short_spread",
    # Robust metrics
    "median_absolute_error",
    "percentile_error",
    "smape",
]
