"""Tests for portfolio allocators.

This module tests:
- EqualWeightAllocator
- InverseVolatilityAllocator
- RiskParityAllocator
- MinimumVarianceAllocator
- MarkowitzAllocator
- RobustMarkowitzAllocator
- MaxSharpeAllocator
- HierarchicalRiskParityAllocator
- CVaRAllocator
"""

import numpy as np
import pandas as pd
import pytest

from src.allocators import (
    AllocatorRegistry,
    CVaRAllocator,
    EqualWeightAllocator,
    HierarchicalRiskParityAllocator,
    InverseVolatilityAllocator,
    MarkowitzAllocator,
    MaxSharpeAllocator,
    MinimumVarianceAllocator,
    RiskParityAllocator,
    RobustMarkowitzAllocator,
)


@pytest.fixture
def sample_returns():
    """Create sample historical returns for multiple assets."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # Create correlated returns for 4 assets
    mean_returns = [0.001, 0.0008, 0.0012, 0.0009]
    volatilities = [0.02, 0.015, 0.025, 0.018]

    returns_dict = {}
    for i, (mu, vol) in enumerate(zip(mean_returns, volatilities, strict=False)):
        returns = np.random.randn(100) * vol + mu
        returns_dict[f"ASSET{i+1}"] = returns

    return pd.DataFrame(returns_dict, index=dates)


@pytest.fixture
def sample_predicted_returns():
    """Create sample predicted returns."""
    return pd.Series(
        [0.01, 0.008, 0.012, 0.009],
        index=["ASSET1", "ASSET2", "ASSET3", "ASSET4"],
    )


class TestEqualWeightAllocator:
    """Test Equal Weight allocator."""

    def test_registration(self):
        """Test that Equal Weight allocator is registered."""
        assert "equal_weight" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("equal_weight")
        assert isinstance(allocator, EqualWeightAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = EqualWeightAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # All weights should be equal
        expected_weight = 1.0 / len(weights)
        for w in weights:
            assert w == pytest.approx(expected_weight)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)

    def test_family(self):
        """Test that Equal Weight has family='simple'."""
        allocator = EqualWeightAllocator()
        assert allocator.family == "simple"

    def test_ignores_predicted_returns(self, sample_predicted_returns, sample_returns):
        """Test that equal weight ignores predicted returns."""
        allocator = EqualWeightAllocator()

        # Different predicted returns should give same weights
        returns1 = pd.Series([0.1, 0.05], index=["A", "B"])
        returns2 = pd.Series([0.01, 0.20], index=["A", "B"])

        hist_returns = pd.DataFrame(
            {"A": np.random.randn(100) * 0.02, "B": np.random.randn(100) * 0.02},
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )

        weights1 = allocator.allocate(returns1, hist_returns)
        weights2 = allocator.allocate(returns2, hist_returns)

        np.testing.assert_array_equal(weights1.values, weights2.values)


class TestInverseVolatilityAllocator:
    """Test Inverse Volatility allocator."""

    def test_registration(self):
        """Test that Inverse Volatility allocator is registered."""
        assert "inverse_volatility" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("inverse_volatility")
        assert isinstance(allocator, InverseVolatilityAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = InverseVolatilityAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)
        # All weights should be positive
        assert all(weights >= 0)

    def test_higher_volatility_lower_weight(self):
        """Test that higher volatility assets get lower weights."""
        # Create returns with different volatilities
        low_vol = np.random.randn(100) * 0.01  # 1% volatility
        high_vol = np.random.randn(100) * 0.05  # 5% volatility

        hist_returns = pd.DataFrame(
            {"LOW": low_vol, "HIGH": high_vol},
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )
        pred_returns = pd.Series([0.01, 0.01], index=["LOW", "HIGH"])

        allocator = InverseVolatilityAllocator()
        weights = allocator.allocate(pred_returns, hist_returns)

        # Low volatility asset should have higher weight
        assert weights["LOW"] > weights["HIGH"]

    def test_family(self):
        """Test that Inverse Volatility has family='simple'."""
        allocator = InverseVolatilityAllocator()
        assert allocator.family == "simple"


class TestRiskParityAllocator:
    """Test Risk Parity allocator."""

    def test_registration(self):
        """Test that Risk Parity allocator is registered."""
        assert "risk_parity" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("risk_parity")
        assert isinstance(allocator, RiskParityAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = RiskParityAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)
        # All weights should be positive
        assert all(weights > 0)

    def test_equal_risk_contribution(self, sample_returns):
        """Test that risk contributions are approximately equal."""
        pred_returns = pd.Series([0.01] * 4, index=sample_returns.columns)

        allocator = RiskParityAllocator()
        weights = allocator.allocate(pred_returns, sample_returns)

        # Compute risk contributions
        cov_matrix = sample_returns.cov()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_vol

        # Risk contributions should be approximately equal
        # Allow some tolerance due to numerical optimization
        assert np.std(risk_contributions) < 0.01

    def test_family(self):
        """Test that Risk Parity has family='risk_based'."""
        allocator = RiskParityAllocator()
        assert allocator.family == "risk_based"


class TestMinimumVarianceAllocator:
    """Test Minimum Variance Portfolio allocator."""

    def test_registration(self):
        """Test that Minimum Variance allocator is registered."""
        assert "minimum_variance" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("minimum_variance")
        assert isinstance(allocator, MinimumVarianceAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = MinimumVarianceAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)
        # Weights should be non-negative (no shorting)
        assert all(weights >= -1e-6)  # Allow small numerical errors

    def test_minimizes_variance(self, sample_returns):
        """Test that minimum variance portfolio has lower variance than equal weight."""
        pred_returns = pd.Series([0.01] * 4, index=sample_returns.columns)

        # Minimum variance portfolio
        mv_allocator = MinimumVarianceAllocator()
        mv_weights = mv_allocator.allocate(pred_returns, sample_returns)

        # Equal weight portfolio
        ew_weights = pd.Series([0.25] * 4, index=sample_returns.columns)

        # Compute portfolio variances
        cov_matrix = sample_returns.cov()
        mv_variance = mv_weights.T @ cov_matrix @ mv_weights
        ew_variance = ew_weights.T @ cov_matrix @ ew_weights

        # Minimum variance should be lower or equal
        assert mv_variance <= ew_variance + 1e-6

    def test_family(self):
        """Test that Minimum Variance has family='risk_based'."""
        allocator = MinimumVarianceAllocator()
        assert allocator.family == "risk_based"


class TestMarkowitzAllocator:
    """Test Markowitz (Mean-Variance) allocator."""

    def test_registration(self):
        """Test that Markowitz allocator is registered."""
        assert "markowitz" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("markowitz")
        assert isinstance(allocator, MarkowitzAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = MarkowitzAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)

    def test_higher_return_higher_weight(self):
        """Test that assets with higher predicted returns get higher weights."""
        np.random.seed(42)
        hist_returns = pd.DataFrame(
            {
                "HIGH": np.random.randn(100) * 0.02 + 0.001,
                "LOW": np.random.randn(100) * 0.02 + 0.001,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )

        # Predict higher return for HIGH
        pred_returns = pd.Series([0.02, 0.005], index=["HIGH", "LOW"])

        allocator = MarkowitzAllocator(config={"risk_aversion": 2.0})
        weights = allocator.allocate(pred_returns, hist_returns)

        # HIGH should have higher weight
        assert weights["HIGH"] > weights["LOW"]

    def test_risk_aversion_parameter(self, sample_predicted_returns, sample_returns):
        """Test effect of risk aversion parameter."""
        # High risk aversion
        high_ra_allocator = MarkowitzAllocator(config={"risk_aversion": 10.0})
        high_ra_weights = high_ra_allocator.allocate(sample_predicted_returns, sample_returns)

        # Low risk aversion
        low_ra_allocator = MarkowitzAllocator(config={"risk_aversion": 0.5})
        low_ra_weights = low_ra_allocator.allocate(sample_predicted_returns, sample_returns)

        # High risk aversion should lead to more conservative (more diversified) portfolio
        # Measure concentration with Herfindahl index
        high_ra_concentration = (high_ra_weights**2).sum()
        low_ra_concentration = (low_ra_weights**2).sum()

        # Lower concentration = more diversified
        assert high_ra_concentration <= low_ra_concentration + 0.1

    def test_family(self):
        """Test that Markowitz has family='mean_variance'."""
        allocator = MarkowitzAllocator()
        assert allocator.family == "mean_variance"


class TestRobustMarkowitzAllocator:
    """Test Robust Markowitz allocator with shrinkage."""

    def test_registration(self):
        """Test that Robust Markowitz allocator is registered."""
        assert "robust_markowitz" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("robust_markowitz")
        assert isinstance(allocator, RobustMarkowitzAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = RobustMarkowitzAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)

    def test_covariance_shrinkage(self, sample_predicted_returns, sample_returns):
        """Test effect of covariance shrinkage."""
        # With shrinkage
        shrunk_allocator = RobustMarkowitzAllocator(config={"covariance_shrinkage": 0.5})
        shrunk_weights = shrunk_allocator.allocate(sample_predicted_returns, sample_returns)

        # Without shrinkage
        no_shrink_allocator = RobustMarkowitzAllocator(config={"covariance_shrinkage": 0.0})
        no_shrink_weights = no_shrink_allocator.allocate(sample_predicted_returns, sample_returns)

        # Weights should differ
        assert not np.allclose(shrunk_weights, no_shrink_weights)

    def test_return_shrinkage(self, sample_returns):
        """Test effect of return shrinkage."""
        # Extreme predicted returns
        extreme_returns = pd.Series([0.5, -0.3, 0.2, -0.1], index=sample_returns.columns)

        # With return shrinkage
        shrunk_allocator = RobustMarkowitzAllocator(config={"return_shrinkage": 0.8})
        shrunk_weights = shrunk_allocator.allocate(extreme_returns, sample_returns)

        # Without return shrinkage
        no_shrink_allocator = RobustMarkowitzAllocator(config={"return_shrinkage": 0.0})
        no_shrink_weights = no_shrink_allocator.allocate(extreme_returns, sample_returns)

        # Shrunk weights should be more balanced
        shrunk_concentration = (shrunk_weights**2).sum()
        no_shrink_concentration = (no_shrink_weights**2).sum()

        assert shrunk_concentration < no_shrink_concentration + 0.05

    def test_l2_penalty(self, sample_predicted_returns, sample_returns):
        """Test effect of L2 penalty."""
        # High L2 penalty should push toward equal weights
        high_l2_allocator = RobustMarkowitzAllocator(config={"l2_penalty": 1.0})
        high_l2_weights = high_l2_allocator.allocate(sample_predicted_returns, sample_returns)

        # Weights should be relatively close to equal (0.25)
        assert all(abs(high_l2_weights - 0.25) < 0.15)

    def test_family(self):
        """Test that Robust Markowitz has family='mean_variance'."""
        allocator = RobustMarkowitzAllocator()
        assert allocator.family == "mean_variance"


class TestMaxSharpeAllocator:
    """Test Maximum Sharpe Ratio allocator."""

    def test_registration(self):
        """Test that Max Sharpe allocator is registered."""
        assert "max_sharpe" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("max_sharpe")
        assert isinstance(allocator, MaxSharpeAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = MaxSharpeAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)

    def test_maximizes_sharpe(self, sample_returns):
        """Test that Max Sharpe has higher Sharpe than equal weight."""
        # Create predicted returns with clear winner
        pred_returns = pd.Series([0.02, 0.005, 0.01, 0.008], index=sample_returns.columns)

        # Max Sharpe portfolio
        ms_allocator = MaxSharpeAllocator()
        ms_weights = ms_allocator.allocate(pred_returns, sample_returns)

        # Equal weight portfolio
        ew_weights = pd.Series([0.25] * 4, index=sample_returns.columns)

        # Compute Sharpe ratios
        cov_matrix = sample_returns.cov()

        ms_return = ms_weights @ pred_returns
        ms_vol = np.sqrt(ms_weights.T @ cov_matrix @ ms_weights)
        ms_sharpe = ms_return / ms_vol

        ew_return = ew_weights @ pred_returns
        ew_vol = np.sqrt(ew_weights.T @ cov_matrix @ ew_weights)
        ew_sharpe = ew_return / ew_vol

        # Max Sharpe should have higher Sharpe ratio
        assert ms_sharpe >= ew_sharpe - 1e-6

    def test_risk_free_rate(self, sample_predicted_returns, sample_returns):
        """Test effect of risk-free rate."""
        # With higher risk-free rate
        high_rf_allocator = MaxSharpeAllocator(config={"risk_free_rate": 0.05})
        high_rf_weights = high_rf_allocator.allocate(sample_predicted_returns, sample_returns)

        # With lower risk-free rate
        low_rf_allocator = MaxSharpeAllocator(config={"risk_free_rate": 0.0})
        low_rf_weights = low_rf_allocator.allocate(sample_predicted_returns, sample_returns)

        # Weights should differ
        assert not np.allclose(high_rf_weights, low_rf_weights)

    def test_family(self):
        """Test that Max Sharpe has family='mean_variance'."""
        allocator = MaxSharpeAllocator()
        assert allocator.family == "mean_variance"


class TestAllocatorComparison:
    """Test comparison of allocators."""

    def test_all_allocators_work_on_same_data(self, sample_predicted_returns, sample_returns):
        """Test that all allocators can allocate on same data."""
        allocators = [
            "equal_weight",
            "inverse_volatility",
            "risk_parity",
            "minimum_variance",
            "markowitz",
            "robust_markowitz",
            "max_sharpe",
        ]

        for allocator_name in allocators:
            allocator = AllocatorRegistry.create(allocator_name)
            weights = allocator.allocate(sample_predicted_returns, sample_returns)

            assert isinstance(weights, pd.Series)
            assert len(weights) == len(sample_predicted_returns)
            assert weights.sum() == pytest.approx(1.0)
            # No shorting (all weights >= 0)
            assert all(weights >= -1e-6)

    def test_weights_sum_to_one(self, sample_predicted_returns, sample_returns):
        """Test that all allocators produce weights that sum to 1."""
        allocators = [
            "equal_weight",
            "inverse_volatility",
            "risk_parity",
            "minimum_variance",
            "markowitz",
            "robust_markowitz",
            "max_sharpe",
        ]

        for allocator_name in allocators:
            allocator = AllocatorRegistry.create(allocator_name)
            weights = allocator.allocate(sample_predicted_returns, sample_returns)
            assert weights.sum() == pytest.approx(1.0)


class TestHierarchicalRiskParityAllocator:
    """Test Hierarchical Risk Parity (HRP) allocator."""

    def test_registration(self):
        """Test that HRP allocator is registered."""
        assert "hrp" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("hrp")
        assert isinstance(allocator, HierarchicalRiskParityAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = HierarchicalRiskParityAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)
        # All weights should be non-negative
        assert all(weights >= 0)

    def test_family(self):
        """Test that HRP has family='advanced'."""
        allocator = HierarchicalRiskParityAllocator()
        assert allocator.family == "advanced"

    def test_no_matrix_inversion_needed(self, sample_predicted_returns, sample_returns):
        """Test that HRP works even with near-singular covariance matrix."""
        # Create returns with some identical columns (singular covariance)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        base_returns = np.random.randn(100) * 0.02
        returns_df = pd.DataFrame(
            {
                "A": base_returns,
                "B": base_returns + np.random.randn(100) * 0.001,  # Almost identical
                "C": np.random.randn(100) * 0.02,
            },
            index=dates,
        )
        pred_returns = pd.Series([0.01, 0.01, 0.01], index=["A", "B", "C"])

        allocator = HierarchicalRiskParityAllocator()
        weights = allocator.allocate(pred_returns, returns_df)

        # Should still work
        assert weights.sum() == pytest.approx(1.0)
        assert all(weights >= 0)

    def test_linkage_methods(self, sample_predicted_returns, sample_returns):
        """Test different hierarchical clustering methods."""
        for method in ["single", "complete", "average"]:
            allocator = HierarchicalRiskParityAllocator(config={"linkage": method})
            weights = allocator.allocate(sample_predicted_returns, sample_returns)
            assert weights.sum() == pytest.approx(1.0)


class TestCVaRAllocator:
    """Test Conditional Value at Risk (CVaR) allocator."""

    def test_registration(self):
        """Test that CVaR allocator is registered."""
        assert "cvar" in AllocatorRegistry.list_all()
        allocator = AllocatorRegistry.create("cvar")
        assert isinstance(allocator, CVaRAllocator)

    def test_allocate(self, sample_predicted_returns, sample_returns):
        """Test allocation."""
        allocator = CVaRAllocator()
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_predicted_returns)
        # Weights should sum to 1
        assert weights.sum() == pytest.approx(1.0)
        # All weights should be non-negative
        assert all(weights >= 0)

    def test_family(self):
        """Test that CVaR has family='advanced'."""
        allocator = CVaRAllocator()
        assert allocator.family == "advanced"

    def test_alpha_parameter(self, sample_predicted_returns, sample_returns):
        """Test effect of alpha (confidence level) parameter."""
        # High alpha = focus on more extreme tail
        high_alpha_allocator = CVaRAllocator(config={"alpha": 0.99})
        high_alpha_weights = high_alpha_allocator.allocate(sample_predicted_returns, sample_returns)

        # Lower alpha = less extreme tail
        low_alpha_allocator = CVaRAllocator(config={"alpha": 0.90})
        low_alpha_weights = low_alpha_allocator.allocate(sample_predicted_returns, sample_returns)

        # Both should produce valid weights
        assert np.isclose(high_alpha_weights.sum(), 1.0)
        assert np.isclose(low_alpha_weights.sum(), 1.0)

    def test_weight_constraints(self, sample_predicted_returns, sample_returns):
        """Test weight constraints."""
        allocator = CVaRAllocator(config={"min_weight": 0.1, "max_weight": 0.5})
        weights = allocator.allocate(sample_predicted_returns, sample_returns)

        # Check constraints (with small tolerance)
        assert all(weights >= 0.1 - 1e-6)
        assert all(weights <= 0.5 + 1e-6)
        assert weights.sum() == pytest.approx(1.0)


class TestAdvancedAllocatorsComparison:
    """Test comparison of all advanced allocators."""

    def test_all_advanced_allocators_work(self, sample_predicted_returns, sample_returns):
        """Test that all advanced allocators produce valid weights."""
        advanced_allocators = [
            "black_litterman",
            "hrp",
            "cvar",
        ]

        for allocator_name in advanced_allocators:
            allocator = AllocatorRegistry.create(allocator_name)
            weights = allocator.allocate(sample_predicted_returns, sample_returns)

            assert isinstance(weights, pd.Series), f"{allocator_name} failed"
            assert len(weights) == len(sample_predicted_returns), f"{allocator_name} wrong length"
            assert weights.sum() == pytest.approx(1.0), f"{allocator_name} doesn't sum to 1"
            assert all(weights >= -1e-6), f"{allocator_name} has negative weights"
            assert allocator.family == "advanced", f"{allocator_name} wrong family"
