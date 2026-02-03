"""
Portfolio Allocators - Consolidated Implementation

All portfolio allocation methods in one place:
- Simple: Equal Weight, Inverse Volatility
- Risk-Based: Risk Parity, Minimum Variance, Maximum Diversification
- Mean-Variance: Markowitz, Robust Markowitz, Max Sharpe
- Advanced: Black-Litterman

Research-backed implementations for production-grade portfolio optimization.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseAllocator
from .registry import register_allocator

logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE ALLOCATORS
# =============================================================================


@register_allocator("equal_weight")
class EqualWeightAllocator(BaseAllocator):
    """
    Equal Weight (1/N) - Maximum diversification baseline.
    Research shows this often beats complex optimizers out-of-sample.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "simple"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        weight = 1.0 / n_assets
        return self._dict_to_series({ticker: weight for ticker in tickers})


@register_allocator("inverse_volatility")
class InverseVolatilityAllocator(BaseAllocator):
    """
    Inverse Volatility - Weights inversely proportional to volatility.
    Config: lookback (60), min_weight (0.0), max_weight (1.0)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "simple"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        lookback = self.config.get("lookback", 60)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)

        tickers = predicted_returns.index.tolist()
        recent_returns = historical_returns[tickers].tail(lookback)
        volatilities = recent_returns.std().replace(0, recent_returns.std().mean())

        inverse_vol = 1.0 / volatilities
        weights_raw = inverse_vol / inverse_vol.sum()
        weights = {ticker: weights_raw[ticker] for ticker in tickers}
        weights = self._apply_constraints(weights, min_weight=min_weight, max_weight=max_weight)
        return self._dict_to_series(weights)


# =============================================================================
# RISK-BASED ALLOCATORS
# =============================================================================


@register_allocator("risk_parity")
class RiskParityAllocator(BaseAllocator):
    """
    Risk Parity - Equal risk contribution from each asset.
    Config: min_weight (0.01), max_weight (1.0), lookback (252)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "risk_based"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        min_weight = self.config.get("min_weight", 0.01)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values

        def risk_parity_objective(weights):
            portfolio_var = weights @ cov_matrix @ weights
            portfolio_std = np.sqrt(portfolio_var)
            marginal_contrib = (cov_matrix @ weights) / (portfolio_std + 1e-8)
            risk_contrib = weights * marginal_contrib
            target = portfolio_std / n_assets
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                risk_parity_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Risk parity failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("minimum_variance")
class MinimumVarianceAllocator(BaseAllocator):
    """
    Minimum Variance - Pure risk minimization (ignores returns).
    Config: min_weight (0.0), max_weight (1.0), lookback (252), regularization (0.0)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "risk_based"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)
        regularization = self.config.get("regularization", 0.0)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values

        if regularization > 0:
            target = np.diag(np.diag(cov_matrix).mean() * np.ones(n_assets))
            cov_matrix = (1 - regularization) * cov_matrix + regularization * target

        def objective(weights):
            return weights @ cov_matrix @ weights

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Minimum variance failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("maximum_diversification")
class MaximumDiversificationAllocator(BaseAllocator):
    """
    Maximum Diversification - Maximizes diversification ratio.

    Diversification Ratio = (Σ w_i * σ_i) / σ_portfolio
    Maximizing this ratio creates a well-diversified portfolio.

    Config:
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        lookback: Days for estimation (default: 252)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "risk_based"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values
        volatilities = np.sqrt(np.diag(cov_matrix))

        def negative_diversification_ratio(weights):
            """Minimize negative diversification ratio = maximize diversification."""
            weighted_vol_sum = np.dot(weights, volatilities)
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            diversification_ratio = weighted_vol_sum / (portfolio_vol + 1e-8)
            return -diversification_ratio  # Minimize negative = maximize

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                negative_diversification_ratio,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                div_ratio = -result.fun
                logger.debug(f"Maximum diversification successful: ratio={div_ratio:.3f}")
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Maximum diversification failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


# =============================================================================
# MEAN-VARIANCE ALLOCATORS
# =============================================================================


@register_allocator("markowitz")
class MarkowitzAllocator(BaseAllocator):
    """
    Classic Mean-Variance (Markowitz) optimization.
    WARNING: Very sensitive to forecast errors. Use RobustMarkowitz instead.
    Config: risk_aversion (1.0), min_weight (0.0), max_weight (1.0), lookback (252)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "mean_variance"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        risk_aversion = self.config.get("risk_aversion", 1.0)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        mu = predicted_returns[tickers].values
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values

        def objective(weights):
            return -np.dot(mu, weights) + risk_aversion * (weights @ cov_matrix @ weights)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Markowitz failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("robust_markowitz")
class RobustMarkowitzAllocator(BaseAllocator):
    """
    Robust Mean-Variance with shrinkage and regularization.
    Much more stable than classic Markowitz

    Config:
        risk_aversion: Risk aversion λ (default: 1.0)
        min_weight, max_weight: Weight bounds (default: 0.0, 1.0)
        covariance_shrinkage: Shrinkage for Σ (default: 0.1)
        return_shrinkage: Shrinkage for μ (default: 0.2)
        regularization: L2 penalty (default: 0.01)
        lookback: Days (default: 252)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "mean_variance"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        # Validate inputs
        self._validate_inputs(predicted_returns, historical_returns, current_prices)

        risk_aversion = self.config.get("risk_aversion", 1.0)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        cov_shrinkage = self.config.get("covariance_shrinkage", 0.1)
        ret_shrinkage = self.config.get("return_shrinkage", 0.2)
        regularization = self.config.get("regularization", 0.01)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)

        # Handle single asset case
        if n_assets == 1:
            return self._dict_to_series({tickers[0]: 1.0})

        # Get recent returns and handle NaN values
        recent_returns = historical_returns[tickers].tail(lookback).dropna()

        # Check sufficient data for covariance estimation
        min_required = max(n_assets + 1, 30)  # At least n+1 observations or 30 days
        if len(recent_returns) < min_required:
            logger.warning(
                f"Insufficient data for robust estimation: {len(recent_returns)} < {min_required}, "
                "using equal weight"
            )
            return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})

        # Shrink covariance matrix
        cov_sample = recent_returns.cov().values
        std_devs = np.sqrt(np.diag(cov_sample))
        # avg_cov is the average off-diagonal covariance (safe: n_assets >= 2 guaranteed)
        avg_cov = (np.sum(cov_sample) - np.sum(np.diag(cov_sample))) / (n_assets * (n_assets - 1))
        cov_target = np.diag(std_devs**2)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    cov_target[i, j] = avg_cov
        cov_matrix = (1 - cov_shrinkage) * cov_sample + cov_shrinkage * cov_target

        # Shrink expected returns
        mu_forecast = predicted_returns[tickers].values
        mu_historical = recent_returns.mean().values
        mu = (1 - ret_shrinkage) * mu_forecast + ret_shrinkage * mu_historical

        w_equal = np.ones(n_assets) / n_assets

        def objective(weights):
            expected_return = np.dot(mu, weights)
            variance = weights @ cov_matrix @ weights
            reg_penalty = regularization * np.sum((weights - w_equal) ** 2)
            return -expected_return + risk_aversion * variance + reg_penalty

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = w_equal.copy()

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Robust Markowitz failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("max_sharpe")
class MaxSharpeAllocator(BaseAllocator):
    """
    Maximum Sharpe Ratio - Tangency portfolio.
    Config: risk_free_rate (0.02), min_weight (0.0), max_weight (1.0), lookback (252)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "mean_variance"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        risk_free_rate = self.config.get("risk_free_rate", 0.02)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        mu = predicted_returns[tickers].values
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values
        rf_daily = risk_free_rate / 252

        def objective(weights):
            port_return = np.dot(mu, weights)
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = (port_return - rf_daily) / (port_vol + 1e-8)
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Max Sharpe failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


# =============================================================================
# ADVANCED ALLOCATORS
# =============================================================================


@register_allocator("black_litterman")
class BlackLittermanAllocator(BaseAllocator):
    """
    Black-Litterman Model - Combines market equilibrium with investor views.

    The Black-Litterman approach:
    1. Start with market equilibrium returns (reverse optimization)
    2. Blend with investor views (forecasts) using Bayesian updating
    3. Optimize portfolio with blended returns

    This reduces sensitivity to return forecasts compared to classic Markowitz.

    Config:
        risk_aversion: Risk aversion parameter (default: 2.5)
        tau: Uncertainty in prior (default: 0.025)
        min_weight, max_weight: Weight bounds (default: 0.0, 1.0)
        lookback: Days for estimation (default: 252)
        view_confidence: Confidence in views (default: 0.25, range 0-1)
            - 0 = ignore views (use market equilibrium)
            - 1 = full confidence in views
        market_weights: Optional dict of market cap weights {ticker: weight}
            If not provided, uses equal weight as market portfolio.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        risk_aversion = self.config.get("risk_aversion", 2.5)
        tau = self.config.get("tau", 0.025)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)
        # Clip view_confidence to prevent division by zero (valid range: 1e-6 to 1.0)
        view_confidence = np.clip(self.config.get("view_confidence", 0.25), 1e-6, 1.0)
        market_weights_config = self.config.get("market_weights", None)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values

        # Step 1: Calculate market equilibrium returns (reverse optimization)
        # Use external market weights if provided, otherwise equal weight
        if market_weights_config is not None:
            market_weights = np.array(
                [market_weights_config.get(t, 1.0 / n_assets) for t in tickers]
            )
            market_weights = market_weights / market_weights.sum()  # Normalize
        else:
            market_weights = np.ones(n_assets) / n_assets
        # Implied returns: Π = λ * Σ * w_market
        pi = risk_aversion * (cov_matrix @ market_weights)

        # Step 2: Process investor views
        # P matrix: identity (each view is about one asset)
        # Q vector: our forecasted returns
        P = np.eye(n_assets)
        Q = predicted_returns[tickers].values

        # Omega: uncertainty in views (diagonal matrix)
        # Higher view_confidence = lower uncertainty
        omega_diag = np.diag(cov_matrix) * tau / view_confidence
        Omega = np.diag(omega_diag)

        # Step 3: Black-Litterman formula - combine prior and views
        # Posterior returns: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
        # Use pseudo-inverse as fallback for singular matrices
        try:
            tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman: tau*cov_matrix singular, using pseudo-inverse")
            tau_sigma_inv = np.linalg.pinv(tau * cov_matrix)

        try:
            omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman: Omega singular, using pseudo-inverse")
            omega_inv = np.linalg.pinv(Omega)

        # Combined precision matrix
        precision = tau_sigma_inv + P.T @ omega_inv @ P

        # Combined mean
        mean_component = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

        # Black-Litterman expected returns
        try:
            bl_returns = np.linalg.solve(precision, mean_component)
        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman matrix inversion failed, using forecasts")
            bl_returns = Q

        # Step 4: Optimize with Black-Litterman returns
        def objective(weights):
            return -np.dot(bl_returns, weights) + risk_aversion * (weights @ cov_matrix @ weights)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = market_weights

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = {ticker: float(w) for ticker, w in zip(tickers, result.x, strict=False)}
                logger.debug(f"Black-Litterman optimization successful: {len(tickers)} assets")
                return self._dict_to_series(self._normalize_weights(weights))
        except Exception as e:
            logger.warning(f"Black-Litterman optimization failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("hrp")
class HierarchicalRiskParityAllocator(BaseAllocator):
    """
    Hierarchical Risk Parity (HRP) - Modern tree-based portfolio construction.

    HRP uses hierarchical clustering to group similar assets and allocates
    weights through recursive bisection. Key advantages:
    - No covariance matrix inversion required (more stable)
    - Naturally handles correlated assets
    - Typically more robust out-of-sample than mean-variance

    Algorithm:
    1. Compute distance matrix from correlations
    2. Hierarchical clustering (single/complete linkage)
    3. Quasi-diagonalize covariance matrix
    4. Recursive bisection allocation

    Config:
        lookback: Days for estimation (default: 252)
        linkage: Clustering method ("single", "complete", "average") (default: "single")
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)

    Reference: López de Prado (2016) - Building Diversified Portfolios that Outperform Out of Sample
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        from scipy.spatial.distance import squareform

        lookback = self.config.get("lookback", 252)
        linkage_method = self.config.get("linkage", "single")
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov()
        corr_matrix = recent_returns.corr()

        # Step 1: Distance matrix from correlations
        # Distance = sqrt(0.5 * (1 - correlation))
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix.values))
        np.fill_diagonal(dist_matrix, 0)

        # Step 2: Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix)
            link = scipy_linkage(condensed_dist, method=linkage_method)
        except Exception as e:
            logger.warning(f"HRP clustering failed: {e}, using equal weight")
            return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})

        # Step 3: Get quasi-diagonal order
        sorted_idx = self._get_quasi_diag(link, n_assets)

        # Step 4: Recursive bisection
        weights = self._recursive_bisection(cov_matrix.values, sorted_idx)

        # Map back to tickers
        weight_dict = {tickers[i]: weights[i] for i in range(n_assets)}

        # Apply constraints
        weight_dict = self._apply_constraints(
            weight_dict, min_weight=min_weight, max_weight=max_weight
        )

        logger.debug(f"HRP allocation successful: {n_assets} assets")
        return self._dict_to_series(weight_dict)

    def _get_quasi_diag(self, link: np.ndarray, n_assets: int) -> list[int]:
        """
        Get quasi-diagonal order from hierarchical clustering (pure function).

        Uses iterative traversal with a stack instead of mutable state.

        :param link: Linkage matrix from scipy hierarchical clustering
        :param n_assets: Number of original assets
        :return: List of asset indices in quasi-diagonal order
        """
        # link is (n_assets-1) x 4 matrix: [cluster1, cluster2, distance, size]
        # Cluster indices: 0 to n_assets-1 are original assets (leaves)
        # Cluster indices: n_assets to 2*n_assets-2 are merged clusters
        # Root cluster (final merge) is at index 2*n_assets-2

        sorted_idx = []
        nodes_to_process = [2 * n_assets - 2]  # Start with root

        while nodes_to_process:
            node = nodes_to_process.pop()
            if node < n_assets:
                # Leaf node (original asset)
                sorted_idx.append(int(node))
            else:
                # Internal node - add children to stack
                row_idx = int(node - n_assets)
                left = int(link[row_idx, 0])
                right = int(link[row_idx, 1])
                # Add right first (LIFO), so left is processed first
                nodes_to_process.append(right)
                nodes_to_process.append(left)

        return sorted_idx if sorted_idx else list(range(n_assets))

    def _recursive_bisection(self, cov: np.ndarray, sorted_idx: list[int]) -> np.ndarray:
        """Allocate weights using recursive bisection."""
        n = len(sorted_idx)
        weights = np.ones(cov.shape[0])

        clusters = [sorted_idx]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue

                # Split cluster in half
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]

                # Calculate cluster variances
                left_var = self._cluster_variance(cov, left_cluster)
                right_var = self._cluster_variance(cov, right_cluster)

                # Allocate based on inverse variance
                total_var = left_var + right_var
                if total_var > 0:
                    alpha = 1 - left_var / total_var
                else:
                    alpha = 0.5

                # Update weights
                for idx in left_cluster:
                    weights[idx] *= alpha
                for idx in right_cluster:
                    weights[idx] *= 1 - alpha

                new_clusters.append(left_cluster)
                new_clusters.append(right_cluster)

            clusters = [c for c in new_clusters if len(c) > 1]

        # Normalize
        weights = weights / weights.sum()
        return weights

    def _cluster_variance(self, cov: np.ndarray, indices: list[int]) -> float:
        """Calculate variance of a cluster (inverse-variance weighted)."""
        if len(indices) == 0:
            return 0.0

        cluster_cov = cov[np.ix_(indices, indices)]
        # Protect against zero variance (clip to small positive value)
        diag_values = np.clip(np.diag(cluster_cov), 1e-12, None)
        ivp_weights = 1.0 / diag_values
        ivp_weights = ivp_weights / ivp_weights.sum()
        variance = float(ivp_weights @ cluster_cov @ ivp_weights)
        return variance


@register_allocator("cvar")
class CVaRAllocator(BaseAllocator):
    """
    CVaR (Conditional Value at Risk) / Expected Shortfall Optimization.

    Minimizes the expected loss in the worst α% of scenarios (tail risk).
    More robust than variance-based methods for extreme events.

    CVaR is:
    - Coherent risk measure (unlike VaR)
    - Better captures tail risk
    - Convex optimization problem (tractable)

    Config:
        alpha: Confidence level (default: 0.95, meaning worst 5% of scenarios)
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        lookback: Days for estimation (default: 252)
        target_return: Optional minimum expected return (default: None)

    Reference: Rockafellar & Uryasev (2000) - Optimization of Conditional Value-at-Risk
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)

        alpha = self.config.get("alpha", 0.95)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)
        target_return = self.config.get("target_return", None)
        allow_short = self.config.get("allow_short", False)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        mu = predicted_returns[tickers].values  # For target_return constraint
        recent_returns = historical_returns[tickers].tail(lookback).values
        n_scenarios = len(recent_returns)

        # CVaR optimization using linear programming formulation
        # min VaR + (1/(1-alpha)) * E[max(0, -r_p - VaR)]
        # This can be linearized using auxiliary variables

        try:
            # Use scipy.optimize for CVaR
            from scipy.optimize import linprog

            # Variables: [weights (n), VaR (1), u (n_scenarios)]
            # u_s >= 0, u_s >= -sum(w_i * r_{s,i}) - VaR
            # min: VaR + (1/((1-alpha)*n_scenarios)) * sum(u_s)

            n_vars = n_assets + 1 + n_scenarios  # weights + VaR + auxiliary u

            # Objective: minimize VaR + (1/((1-alpha)*n_scenarios)) * sum(u)
            c = np.zeros(n_vars)
            c[n_assets] = 1  # VaR coefficient
            c[n_assets + 1 :] = 1 / ((1 - alpha) * n_scenarios)  # u coefficients

            # Inequality constraints: -sum(w_i * r_{s,i}) - VaR - u_s <= 0
            # Rewritten: sum(w_i * r_{s,i}) + VaR + u_s >= 0
            A_ub = np.zeros((n_scenarios, n_vars))
            for s in range(n_scenarios):
                A_ub[s, :n_assets] = -recent_returns[s]  # -returns for each scenario
                A_ub[s, n_assets] = -1  # -VaR
                A_ub[s, n_assets + 1 + s] = -1  # -u_s

            b_ub = np.zeros(n_scenarios)

            # Add target_return constraint if specified: mu @ w >= target_return
            # Rewritten as: -mu @ w <= -target_return
            if target_return is not None:
                A_return = np.zeros((1, n_vars))
                A_return[0, :n_assets] = -mu  # -mu @ w
                b_return = np.array([-target_return])
                A_ub = np.vstack([A_ub, A_return])
                b_ub = np.concatenate([b_ub, b_return])

            # Equality constraint: sum of weights = 1
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n_assets] = 1
            b_eq = np.array([1.0])

            # Bounds: weights in [min_weight, max_weight], VaR unbounded, u >= 0
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            bounds.append((None, None))  # VaR can be any value
            bounds.extend([(0, None) for _ in range(n_scenarios)])  # u >= 0

            # Solve
            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
            )

            if result.success:
                weights = result.x[:n_assets]
                # Only force non-negative if allow_short=False
                if not allow_short:
                    weights = np.maximum(weights, 0)
                weights = weights / weights.sum()  # Normalize
                weight_dict = {
                    ticker: float(w) for ticker, w in zip(tickers, weights, strict=False)
                }
                logger.debug(f"CVaR optimization successful: {n_assets} assets")
                return self._dict_to_series(weight_dict)

        except Exception as e:
            logger.warning(f"CVaR optimization failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("kelly_criterion")
class KellyCriterionAllocator(BaseAllocator):
    """
    Kelly Criterion - Optimal bet sizing for maximum long-term growth.

    The Kelly Criterion maximizes the expected logarithm of wealth (geometric growth).
    Often produces aggressive allocations, so fractional Kelly (half-Kelly) is common.

    Formula: f* = (μ - rf) / σ² for single asset
    For multiple assets: max E[log(1 + r_p)]

    WARNING: Full Kelly can be very aggressive. Use fraction < 1.0 in practice.

    Config:
        fraction: Kelly fraction (0.5 = half-Kelly, recommended) (default: 0.5)
        risk_free_rate: Annual risk-free rate (default: 0.02)
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        lookback: Days for estimation (default: 252)
        use_log_returns: Use log returns for calculation (default: True)

    Reference: Kelly, J.L. (1956) - A New Interpretation of Information Rate
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)
        fraction = self.config.get("fraction", 0.5)  # Half-Kelly by default
        risk_free_rate = self.config.get("risk_free_rate", 0.02)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback)
        cov_matrix = recent_returns.cov().values
        rf_daily = risk_free_rate / 252

        # Kelly formula: f* = Σ^(-1) * (μ - rf)
        # For multiple assets, this is the tangency portfolio scaled
        mu = predicted_returns[tickers].values - rf_daily

        try:
            # Regularized inverse for stability
            cov_matrix_reg = cov_matrix + np.eye(n_assets) * 1e-6
            cov_inv = np.linalg.inv(cov_matrix_reg)

            # Unconstrained Kelly weights
            kelly_weights = cov_inv @ mu

            # Scale by fraction
            kelly_weights = kelly_weights * fraction

            # Normalize to sum to 1 (or less if short positions)
            if kelly_weights.sum() > 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            else:
                # If all negative, use equal weight
                kelly_weights = np.ones(n_assets) / n_assets

            # Apply constraints
            kelly_weights = np.clip(kelly_weights, min_weight, max_weight)
            kelly_weights = kelly_weights / kelly_weights.sum()

            weight_dict = {
                ticker: float(w) for ticker, w in zip(tickers, kelly_weights, strict=False)
            }
            logger.debug(f"Kelly Criterion allocation: fraction={fraction}")
            return self._dict_to_series(weight_dict)

        except Exception as e:
            logger.warning(f"Kelly Criterion failed: {e}, using equal weight")
            return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("mean_cvar")
class MeanCVaRAllocator(BaseAllocator):
    """
    Mean-CVaR Optimization - Balance return and tail risk.

    Maximizes expected return subject to a CVaR (tail risk) constraint.
    This is the mean-risk efficient frontier using CVaR instead of variance.

    Advantages over Mean-Variance:
    - Better handles fat-tailed distributions
    - More intuitive risk measure (expected loss in worst cases)
    - Coherent risk measure (unlike VaR)

    Config:
        target_cvar: Maximum acceptable CVaR (default: 0.05 = 5% expected loss)
        alpha: CVaR confidence level (default: 0.95)
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        lookback: Days for estimation (default: 252)
        return_weight: Weight on expected return in objective (default: 1.0)

    Reference: Rockafellar & Uryasev (2000)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)

        target_cvar = self.config.get("target_cvar", 0.05)
        alpha = self.config.get("alpha", 0.95)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)
        return_weight = self.config.get("return_weight", 1.0)
        allow_short = self.config.get("allow_short", False)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback).values
        n_scenarios = len(recent_returns)
        mu = predicted_returns[tickers].values

        try:
            from scipy.optimize import minimize

            # Optimization variables: weights
            def objective(weights):
                """Maximize expected return (minimize negative return)."""
                return -return_weight * np.dot(mu, weights)

            def cvar_constraint(weights):
                """Compute CVaR constraint: CVaR <= target_cvar."""
                portfolio_returns = recent_returns @ weights
                sorted_returns = np.sort(portfolio_returns)
                var_idx = int(np.floor((1 - alpha) * n_scenarios))
                var_idx = max(1, min(var_idx, n_scenarios - 1))
                cvar = -np.mean(sorted_returns[:var_idx])
                return target_cvar - cvar  # >= 0

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": cvar_constraint},
            ]
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            x0 = np.ones(n_assets) / n_assets

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500},
            )

            if result.success:
                weights = result.x
                # Only force non-negative if allow_short=False
                if not allow_short:
                    weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                weight_dict = {
                    ticker: float(w) for ticker, w in zip(tickers, weights, strict=False)
                }
                logger.debug(f"Mean-CVaR optimization successful: target_cvar={target_cvar}")
                return self._dict_to_series(weight_dict)

        except Exception as e:
            logger.warning(f"Mean-CVaR optimization failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


@register_allocator("omega_ratio")
class OmegaRatioAllocator(BaseAllocator):
    """
    Omega Ratio Optimization - Maximize risk-adjusted returns including all moments.

    The Omega ratio is the probability-weighted ratio of gains versus losses
    relative to a threshold (usually risk-free rate). Unlike Sharpe ratio,
    it considers the entire return distribution, not just mean and variance.

    Omega = ∫[threshold to ∞] (1 - F(r)) dr / ∫[-∞ to threshold] F(r) dr

    Where F(r) is the cumulative distribution function of returns.

    Advantages:
    - Captures skewness and kurtosis naturally
    - More appropriate for non-normal distributions
    - Consistent with first-order stochastic dominance

    Config:
        threshold: Return threshold (default: 0.0 = break-even)
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        lookback: Days for estimation (default: 252)

    Reference: Keating & Shadwick (2002) - A Universal Performance Measure
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.family = "advanced"

    def allocate(
        self,
        predicted_returns: pd.Series,
        historical_returns: pd.DataFrame,
        current_prices: pd.Series | None = None,
    ) -> pd.Series:
        self._validate_inputs(predicted_returns, historical_returns, current_prices)

        threshold = self.config.get("threshold", 0.0)
        min_weight = self.config.get("min_weight", 0.0)
        max_weight = self.config.get("max_weight", 1.0)
        lookback = self.config.get("lookback", 252)
        allow_short = self.config.get("allow_short", False)

        tickers = predicted_returns.index.tolist()
        n_assets = len(tickers)
        recent_returns = historical_returns[tickers].tail(lookback).values
        n_scenarios = len(recent_returns)

        def negative_omega_ratio(weights):
            """Minimize negative Omega ratio = maximize Omega ratio."""
            portfolio_returns = recent_returns @ weights

            gains = portfolio_returns[portfolio_returns > threshold] - threshold
            losses = threshold - portfolio_returns[portfolio_returns <= threshold]

            expected_gain = np.sum(gains) / n_scenarios if len(gains) > 0 else 0
            expected_loss = np.sum(losses) / n_scenarios if len(losses) > 0 else 1e-8

            omega = expected_gain / (expected_loss + 1e-8)
            return -omega

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                negative_omega_ratio,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                weights = result.x
                # Only force non-negative if allow_short=False
                if not allow_short:
                    weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                omega_ratio = -result.fun
                weight_dict = {
                    ticker: float(w) for ticker, w in zip(tickers, weights, strict=False)
                }
                logger.debug(f"Omega ratio optimization successful: ratio={omega_ratio:.3f}")
                return self._dict_to_series(weight_dict)

        except Exception as e:
            logger.warning(f"Omega ratio optimization failed: {e}, using equal weight")

        return self._dict_to_series({ticker: 1.0 / n_assets for ticker in tickers})


__all__ = [
    "EqualWeightAllocator",
    "InverseVolatilityAllocator",
    "RiskParityAllocator",
    "MinimumVarianceAllocator",
    "MaximumDiversificationAllocator",
    "MarkowitzAllocator",
    "RobustMarkowitzAllocator",
    "MaxSharpeAllocator",
    "BlackLittermanAllocator",
    "HierarchicalRiskParityAllocator",
    "CVaRAllocator",
    "KellyCriterionAllocator",
    "MeanCVaRAllocator",
    "OmegaRatioAllocator",
]
