"""
Allocators Module
=================

Portfolio allocation strategies for optimal weight assignment.

Available allocators:
- Simple: equal_weight, inverse_volatility
- Risk-Based: risk_parity, minimum_variance, maximum_diversification
- Mean-Variance: markowitz, robust_markowitz, max_sharpe
- Advanced: black_litterman, hrp, cvar, kelly_criterion, mean_cvar, omega_ratio

Usage:
    >>> from src.allocators import AllocatorRegistry
    >>> allocator = AllocatorRegistry.create("black_litterman", {
    ...     "risk_aversion": 2.5,
    ...     "view_confidence": 0.3
    ... })
    >>> weights = allocator.allocate(predicted_returns, historical_returns)
"""

from .base import BaseAllocator

# Import all allocators from models.py (triggers registration)
from .models import (
    BlackLittermanAllocator,
    CVaRAllocator,
    EqualWeightAllocator,
    HierarchicalRiskParityAllocator,
    InverseVolatilityAllocator,
    KellyCriterionAllocator,
    MarkowitzAllocator,
    MaximumDiversificationAllocator,
    MaxSharpeAllocator,
    MeanCVaRAllocator,
    MinimumVarianceAllocator,
    OmegaRatioAllocator,
    RiskParityAllocator,
    RobustMarkowitzAllocator,
)
from .registry import AllocatorRegistry, get_allocator, register_allocator

__all__ = [
    "BaseAllocator",
    "AllocatorRegistry",
    "get_allocator",
    "register_allocator",
    # Allocator classes
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
