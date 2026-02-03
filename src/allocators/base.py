"""
Base Allocator Abstract Class
==============================

This module defines the abstract base class for all portfolio allocation methods.
All allocators must inherit from :class:`BaseAllocator` and implement the
:meth:`allocate` method.

Example
-------
>>> from src.allocators import AllocatorRegistry
>>> allocator = AllocatorRegistry.create("risk_parity")
>>> weights = allocator.allocate(predicted_returns, historical_returns)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseAllocator(ABC):
    """
    Abstract base class for portfolio allocators.

    All portfolio allocation methods must inherit from this class and
    implement the :meth:`allocate` method.

    :ivar name: Human-readable name of the allocator (auto-generated from class name)
    :vartype name: str
    :ivar family: Category of allocator (simple, mean_variance, risk_based, advanced)
    :vartype family: str
    :ivar config: Configuration dictionary for allocator parameters
    :vartype config: dict

    Important: Time Scale Consistency
    ---------------------------------
    Returns and risk parameters must be on the same time scale:

    - If ``predicted_returns`` are **daily** returns, then ``risk_free_rate``
      should be daily (annual / 252), and covariance is computed from daily data.
    - If ``predicted_returns`` are **annualized**, then ``risk_free_rate``
      should be annual, and covariance should be annualized (daily_cov * 252).

    For mean-variance optimization (Markowitz, MaxSharpe, Kelly):
    - Daily returns typically range from -0.05 to 0.05
    - Annual risk-free rate of 2% → daily rate ≈ 0.02/252 ≈ 0.00008

    Example
    -------
    >>> class MyAllocator(BaseAllocator):
    ...     family = "custom"
    ...     def allocate(self, predicted_returns, historical_returns, current_prices=None):
    ...         n = len(predicted_returns)
    ...         weights = {ticker: 1/n for ticker in predicted_returns.index}
    ...         return self._dict_to_series(weights)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the allocator.

        :param config: Optional dictionary of allocator configuration/parameters
        :type config: dict, optional
        """
        self.config = config or {}
        self.name = self.__class__.__name__.replace("Allocator", "").lower()
        # Only set family if not already defined as class attribute
        if not hasattr(self.__class__, "family"):
            self.family = "unknown"

    @abstractmethod
    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute portfolio weights.

        :param predicted_returns: Expected returns for each ticker (Series with ticker as index)
        :type predicted_returns: pd.Series
        :param historical_returns: Historical returns DataFrame (dates as index, tickers as columns)
        :type historical_returns: pd.DataFrame
        :param current_prices: Optional current prices for each ticker
        :type current_prices: pd.Series, optional
        :return: Series mapping ticker to weight (weights sum to 1.0)
        :rtype: pd.Series
        :raises ValueError: If inputs are invalid
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Return allocator configuration for logging/serialization.

        :return: Dictionary of allocator parameters and settings
        :rtype: dict
        """
        return {"name": self.name, "family": self.family, **self.config}

    @classmethod
    def get_name(cls) -> str:
        """
        Return human-readable allocator name.

        :return: Allocator name (e.g., "Markowitz", "RiskParity", "EqualWeight")
        :rtype: str
        """
        return cls.__name__.replace("Allocator", "")

    def _validate_inputs(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> None:
        """
        Validate input data format and consistency.

        :param predicted_returns: Expected returns Series
        :type predicted_returns: pd.Series
        :param historical_returns: Historical returns DataFrame
        :type historical_returns: pd.DataFrame
        :param current_prices: Optional current prices Series
        :type current_prices: pd.Series, optional
        :raises ValueError: If inputs are invalid or inconsistent
        """
        # Validate predicted_returns
        if not isinstance(predicted_returns, pd.Series):
            raise ValueError(
                f"predicted_returns must be pandas Series, got {type(predicted_returns)}"
            )

        if len(predicted_returns) == 0:
            raise ValueError("predicted_returns cannot be empty")

        if predicted_returns.isna().any():
            raise ValueError("predicted_returns contains NaN values")

        # Validate historical_returns
        if not isinstance(historical_returns, pd.DataFrame):
            raise ValueError(
                f"historical_returns must be pandas DataFrame, got {type(historical_returns)}"
            )

        if len(historical_returns) == 0:
            raise ValueError("historical_returns cannot be empty")

        if not isinstance(historical_returns.index, pd.DatetimeIndex):
            raise ValueError("historical_returns must have DatetimeIndex")

        # Check ticker consistency
        predicted_tickers = set(predicted_returns.index)
        historical_tickers = set(historical_returns.columns)

        if not predicted_tickers.issubset(historical_tickers):
            missing = predicted_tickers - historical_tickers
            raise ValueError(
                f"Tickers in predicted_returns not found in historical_returns: {missing}"
            )

        # Validate current_prices if provided
        if current_prices is not None:
            if not isinstance(current_prices, pd.Series):
                raise ValueError(
                    f"current_prices must be pandas Series, got {type(current_prices)}"
                )

            if current_prices.isna().any():
                raise ValueError("current_prices contains NaN values")

            prices_tickers = set(current_prices.index)
            if not predicted_tickers.issubset(prices_tickers):
                missing = predicted_tickers - prices_tickers
                raise ValueError(
                    f"Tickers in predicted_returns not found in current_prices: {missing}"
                )

    def _validate_weights(self, weights: dict[str, float]) -> None:
        """
        Validate that weights are valid portfolio allocations.

        :param weights: Dictionary of ticker to weight
        :type weights: dict
        :raises ValueError: If weights are invalid
        """
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")

        # Check for negative weights (unless short selling is allowed)
        if any(w < 0 for w in weights.values()):
            if not self.config.get("allow_short", False):
                raise ValueError("Negative weights detected but short selling not allowed")

        # Check if weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-4):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.6f}. "
                "Check your allocation algorithm."
            )

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """
        Normalize weights to sum to exactly 1.0.

        :param weights: Dictionary of ticker to weight
        :type weights: dict
        :return: Normalized weights dictionary
        :rtype: dict
        :raises ValueError: If weights sum to zero
        """
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Cannot normalize weights: sum is zero")

        return {ticker: w / total for ticker, w in weights.items()}

    def _dict_to_series(self, weights: dict[str, float]) -> pd.Series:
        """
        Convert weights dictionary to pandas Series.

        :param weights: Dictionary of ticker to weight
        :type weights: dict
        :return: Series with tickers as index and weights as values
        :rtype: pd.Series
        """
        return pd.Series(weights)

    def _safe_cov_inverse(
        self,
        cov_matrix: np.ndarray,
        regularization: float = 1e-6,
    ) -> np.ndarray:
        """
        Safely compute covariance matrix inverse with regularization.

        Uses Tikhonov regularization and falls back to pseudo-inverse
        if the matrix is still singular.

        :param cov_matrix: Covariance matrix (n x n)
        :type cov_matrix: np.ndarray
        :param regularization: Regularization factor added to diagonal
        :type regularization: float
        :return: Inverse (or pseudo-inverse) of covariance matrix
        :rtype: np.ndarray
        """
        n = cov_matrix.shape[0]

        # Add regularization to diagonal
        cov_reg = cov_matrix + regularization * np.eye(n)

        try:
            # Try standard inverse first
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            cov_inv = np.linalg.pinv(cov_reg)

        return cov_inv

    def _estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = "sample",
    ) -> np.ndarray:
        """
        Estimate covariance matrix with optional shrinkage.

        :param returns: DataFrame of returns (dates as index, tickers as columns)
        :type returns: pd.DataFrame
        :param method: Estimation method: 'sample' or 'ledoit_wolf'
        :type method: str
        :return: Covariance matrix
        :rtype: np.ndarray

        Note: 'ledoit_wolf' applies shrinkage toward a structured estimator,
        which is more stable when n_samples < n_features or data is noisy.
        """
        if method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf()
                lw.fit(returns.values)
                return lw.covariance_
            except ImportError:
                # sklearn not available, fallback to sample
                return returns.cov().values
        else:
            # Default: sample covariance
            return returns.cov().values

    def _apply_constraints(
        self,
        weights: dict[str, float],
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """
        Apply minimum and maximum weight constraints with iterative projection.

        Uses iterative clipping and normalization to find valid portfolio weights
        that satisfy both bound constraints and sum-to-one constraint.

        :param weights: Dictionary of ticker to weight
        :type weights: dict
        :param min_weight: Minimum weight per asset
        :type min_weight: float
        :param max_weight: Maximum weight per asset
        :type max_weight: float
        :param max_iterations: Maximum projection iterations (default: 100)
        :type max_iterations: int
        :param tolerance: Convergence tolerance (default: 1e-6)
        :type tolerance: float
        :return: Constrained and normalized weights
        :rtype: dict
        """
        n_assets = len(weights)
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])

        # Iterative projection: alternate between clipping and normalizing
        for _ in range(max_iterations):
            # Clip to bounds
            w_clipped = np.clip(w, min_weight, max_weight)

            # Check if sum is valid
            total = w_clipped.sum()
            if total <= 0:
                # Fallback to equal weight
                w = np.ones(n_assets) / n_assets
                break

            # Normalize
            w_new = w_clipped / total

            # Check convergence
            if np.allclose(w, w_new, atol=tolerance):
                w = w_new
                break

            w = w_new

        # Final clip and normalize
        w = np.clip(w, min_weight, max_weight)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.ones(n_assets) / n_assets

        return {t: float(w[i]) for i, t in enumerate(tickers)}

    def __repr__(self) -> str:
        """String representation of allocator."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.get_name()} Allocator"
