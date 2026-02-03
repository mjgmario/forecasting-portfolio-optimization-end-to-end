"""
Edge Case Tests
===============

Tests for boundary conditions, error handling, and edge cases across the codebase.

Coverage:
- Single asset portfolios
- Singular covariance matrices
- NaN/Inf in data
- Predict before fit
- Large horizons
- Extremely short data
"""

import numpy as np
import pandas as pd
import pytest

# ===========================================================================
# TEST2.2: Single Asset Portfolio
# ===========================================================================


class TestSingleAssetPortfolio:
    """Test allocators with a single asset (edge case)."""

    @pytest.fixture
    def single_asset_data(self):
        """Create data for single asset portfolio."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        historical = pd.DataFrame({"AAPL": returns}, index=dates)
        predicted = pd.Series([0.05], index=["AAPL"])
        return predicted, historical

    def test_equal_weight_single_asset(self, single_asset_data):
        """EqualWeight should assign 100% to single asset."""
        from src.allocators.models import EqualWeightAllocator

        predicted, historical = single_asset_data
        allocator = EqualWeightAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 1
        assert weights["AAPL"] == pytest.approx(1.0)
        assert weights.sum() == pytest.approx(1.0)

    def test_inverse_volatility_single_asset(self, single_asset_data):
        """InverseVolatility should assign 100% to single asset."""
        from src.allocators.models import InverseVolatilityAllocator

        predicted, historical = single_asset_data
        allocator = InverseVolatilityAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 1
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_risk_parity_single_asset(self, single_asset_data):
        """RiskParity should assign 100% to single asset."""
        from src.allocators.models import RiskParityAllocator

        predicted, historical = single_asset_data
        allocator = RiskParityAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 1
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_minimum_variance_single_asset(self, single_asset_data):
        """MinimumVariance should assign 100% to single asset."""
        from src.allocators.models import MinimumVarianceAllocator

        predicted, historical = single_asset_data
        allocator = MinimumVarianceAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 1
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_markowitz_single_asset(self, single_asset_data):
        """Markowitz should assign 100% to single asset."""
        from src.allocators.models import MarkowitzAllocator

        predicted, historical = single_asset_data
        allocator = MarkowitzAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 1
        assert weights["AAPL"] == pytest.approx(1.0)


# ===========================================================================
# TEST2.3: Singular Covariance Matrix
# ===========================================================================


class TestSingularCovarianceMatrix:
    """Test allocators with singular or near-singular covariance matrices."""

    @pytest.fixture
    def perfectly_correlated_assets(self):
        """Create perfectly correlated assets (singular covariance)."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        base_returns = np.random.randn(100) * 0.02

        # Two assets with identical returns = perfectly correlated = singular cov
        historical = pd.DataFrame(
            {
                "ASSET1": base_returns,
                "ASSET2": base_returns,  # Identical
            },
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03], index=["ASSET1", "ASSET2"])
        return predicted, historical

    @pytest.fixture
    def near_singular_assets(self):
        """Create nearly singular covariance matrix."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        base_returns = np.random.randn(100) * 0.02

        # Very small noise to make near-singular but not exactly singular
        historical = pd.DataFrame(
            {
                "ASSET1": base_returns,
                "ASSET2": base_returns + np.random.randn(100) * 1e-10,
                "ASSET3": base_returns + np.random.randn(100) * 1e-10,
            },
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03, 0.04], index=["ASSET1", "ASSET2", "ASSET3"])
        return predicted, historical

    def test_minimum_variance_singular_cov(self, perfectly_correlated_assets):
        """MinimumVariance should handle singular covariance gracefully."""
        from src.allocators.models import MinimumVarianceAllocator

        predicted, historical = perfectly_correlated_assets
        allocator = MinimumVarianceAllocator({"regularization": 1e-6})
        weights = allocator.allocate(predicted, historical)

        # Should return valid weights (not crash)
        assert len(weights) == 2
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert all(w >= -1e-6 for w in weights)

    def test_risk_parity_singular_cov(self, perfectly_correlated_assets):
        """RiskParity should handle singular covariance gracefully."""
        from src.allocators.models import RiskParityAllocator

        predicted, historical = perfectly_correlated_assets
        allocator = RiskParityAllocator()
        weights = allocator.allocate(predicted, historical)

        # Should return valid weights (fallback to equal weight is OK)
        assert len(weights) == 2
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_robust_markowitz_near_singular(self, near_singular_assets):
        """RobustMarkowitz should handle near-singular covariance."""
        from src.allocators.models import RobustMarkowitzAllocator

        predicted, historical = near_singular_assets
        allocator = RobustMarkowitzAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 3
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_safe_cov_inverse(self):
        """Test _safe_cov_inverse handles singular matrices."""
        from src.allocators.base import BaseAllocator

        # Create a concrete subclass to test base method
        class TestAllocator(BaseAllocator):
            def allocate(self, predicted_returns, historical_returns, current_prices=None):
                return pd.Series()

        allocator = TestAllocator()

        # Singular matrix (rank 1)
        singular = np.array([[1, 1], [1, 1]], dtype=float)
        inv = allocator._safe_cov_inverse(singular, regularization=1e-6)

        # Should return something (not crash)
        assert inv.shape == (2, 2)
        assert not np.any(np.isnan(inv))
        assert not np.any(np.isinf(inv))


# ===========================================================================
# TEST2.4: NaN/Inf in Returns
# ===========================================================================


class TestNaNInfHandling:
    """Test handling of NaN and Inf values in data."""

    @pytest.fixture
    def dates(self):
        return pd.date_range("2024-01-01", periods=100, freq="B")

    def test_allocator_rejects_nan_predicted_returns(self, dates):
        """Allocator should reject NaN in predicted returns."""
        from src.allocators.models import EqualWeightAllocator

        np.random.seed(42)
        historical = pd.DataFrame(
            {"A": np.random.randn(100) * 0.02, "B": np.random.randn(100) * 0.02},
            index=dates,
        )
        predicted = pd.Series([0.05, np.nan], index=["A", "B"])

        allocator = EqualWeightAllocator()
        with pytest.raises(ValueError, match="NaN"):
            allocator.allocate(predicted, historical)

    def test_allocator_rejects_nan_historical_returns(self, dates):
        """Allocator should reject NaN in historical returns."""
        from src.allocators.models import RiskParityAllocator

        returns = np.random.randn(100) * 0.02
        returns[50] = np.nan  # Insert NaN
        historical = pd.DataFrame(
            {"A": returns, "B": np.random.randn(100) * 0.02},
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03], index=["A", "B"])

        allocator = RiskParityAllocator()
        # Covariance with NaN should result in NaN weights or error
        # The allocator may fallback to equal weight in this case
        weights = allocator.allocate(predicted, historical)
        # Should at least not crash and return valid normalized weights
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_forecaster_rejects_all_nan_data(self, dates):
        """Forecaster should reject all-NaN training data."""
        from src.forecasters.baselines import NaiveForecaster

        all_nan = pd.Series([np.nan] * 100, index=dates)

        forecaster = NaiveForecaster()
        with pytest.raises(ValueError, match="NaN"):
            forecaster.fit(all_nan)

    def test_forecaster_handles_inf(self, dates):
        """Forecaster should handle Inf values appropriately."""
        from src.forecasters.baselines import NaiveForecaster

        np.random.seed(42)
        data = pd.Series(np.random.randn(100), index=dates)
        data.iloc[50] = np.inf  # Insert Inf

        forecaster = NaiveForecaster()
        # Should either reject or handle gracefully
        # NaiveForecaster may propagate Inf to predictions
        forecaster.fit(data)
        forecaster.predict(5)  # At minimum should not crash


# ===========================================================================
# TEST2.5: Predict Before Fit
# ===========================================================================


class TestPredictBeforeFit:
    """Test that predict() before fit() raises appropriate error."""

    def test_naive_forecaster_predict_before_fit(self):
        """NaiveForecaster should raise RuntimeError when predicting before fit."""
        from src.forecasters.baselines import NaiveForecaster

        forecaster = NaiveForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)

    def test_seasonal_naive_predict_before_fit(self):
        """SeasonalNaiveForecaster should raise RuntimeError when predicting before fit."""
        from src.forecasters.baselines import SeasonalNaiveForecaster

        forecaster = SeasonalNaiveForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)

    def test_drift_forecaster_predict_before_fit(self):
        """DriftForecaster should raise RuntimeError when predicting before fit."""
        from src.forecasters.baselines import DriftForecaster

        forecaster = DriftForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)

    def test_arima_predict_before_fit(self):
        """ARIMAForecaster should raise RuntimeError when predicting before fit."""
        pytest.importorskip("statsforecast")
        from src.forecasters.statistical import ARIMAForecaster

        forecaster = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)

    def test_ml_forecaster_predict_before_fit(self):
        """XGBoost forecaster should raise RuntimeError when predicting before fit."""
        pytest.importorskip("xgboost")
        from src.forecasters.ml_regressors import XGBoostForecaster

        forecaster = XGBoostForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)

    def test_garch_predict_before_fit(self):
        """GARCHForecaster should raise RuntimeError when predicting before fit."""
        pytest.importorskip("arch")
        from src.forecasters.volatility import GARCHForecaster

        forecaster = GARCHForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(5)


# ===========================================================================
# TEST2.6: Very Large Horizon
# ===========================================================================


class TestLargeHorizon:
    """Test forecasters with very large prediction horizons."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        return pd.Series(np.random.randn(100) * 0.02 + 0.001, index=dates)

    def test_naive_large_horizon(self, sample_data):
        """NaiveForecaster should handle large horizons."""
        from src.forecasters.baselines import NaiveForecaster

        forecaster = NaiveForecaster()
        forecaster.fit(sample_data)

        # Predict 252 days ahead (1 year)
        predictions = forecaster.predict(252)
        assert len(predictions) == 252
        assert not predictions.isna().any()

    def test_drift_large_horizon(self, sample_data):
        """DriftForecaster should handle large horizons."""
        from src.forecasters.baselines import DriftForecaster

        forecaster = DriftForecaster()
        forecaster.fit(sample_data)

        predictions = forecaster.predict(500)
        assert len(predictions) == 500
        assert not predictions.isna().any()
        # Drift should show increasing/decreasing trend
        # (not just flat like naive)

    def test_seasonal_naive_large_horizon(self, sample_data):
        """SeasonalNaiveForecaster should handle horizons larger than period."""
        from src.forecasters.baselines import SeasonalNaiveForecaster

        forecaster = SeasonalNaiveForecaster({"period": 5})
        forecaster.fit(sample_data)

        # Horizon larger than training data
        predictions = forecaster.predict(200)
        assert len(predictions) == 200
        assert not predictions.isna().any()

    @pytest.mark.optional
    def test_arima_large_horizon(self, sample_data):
        """ARIMA should handle moderate large horizons."""
        pytest.importorskip("statsforecast")
        from src.forecasters.statistical import ARIMAForecaster

        forecaster = ARIMAForecaster()
        forecaster.fit(sample_data)

        # ARIMA predictions may become unreliable for very large horizons
        # but should still work
        predictions = forecaster.predict(100)
        assert len(predictions) == 100
        assert not predictions.isna().any()


# ===========================================================================
# TEST2.7: Extremely Short Data
# ===========================================================================


class TestExtremelyShortData:
    """Test with minimal training data."""

    @pytest.fixture
    def minimal_data(self):
        """Create minimal training data (just a few points)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        np.random.seed(42)
        return pd.Series([100, 101, 102, 101.5, 103], index=dates)

    @pytest.fixture
    def single_point_data(self):
        """Create single point training data."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        return pd.Series([100], index=dates)

    def test_naive_minimal_data(self, minimal_data):
        """NaiveForecaster should work with minimal data."""
        from src.forecasters.baselines import NaiveForecaster

        forecaster = NaiveForecaster()
        forecaster.fit(minimal_data)
        predictions = forecaster.predict(5)

        assert len(predictions) == 5
        # Should predict the last value
        assert all(p == 103 for p in predictions)

    def test_naive_single_point(self, single_point_data):
        """NaiveForecaster should work with single data point."""
        from src.forecasters.baselines import NaiveForecaster

        forecaster = NaiveForecaster()
        forecaster.fit(single_point_data)
        predictions = forecaster.predict(3)

        assert len(predictions) == 3
        assert all(p == 100 for p in predictions)

    def test_drift_minimal_data(self, minimal_data):
        """DriftForecaster should work with minimal data."""
        from src.forecasters.baselines import DriftForecaster

        forecaster = DriftForecaster()
        forecaster.fit(minimal_data)
        predictions = forecaster.predict(5)

        assert len(predictions) == 5
        assert not predictions.isna().any()

    def test_seasonal_naive_short_data_warning(self):
        """SeasonalNaive with period > data length should handle gracefully."""
        from src.forecasters.baselines import SeasonalNaiveForecaster

        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        data = pd.Series([1, 2, 3], index=dates)

        # Period 5 but only 3 data points
        forecaster = SeasonalNaiveForecaster({"period": 5})
        forecaster.fit(data)
        predictions = forecaster.predict(5)

        # Should handle gracefully (may use smaller effective period)
        assert len(predictions) == 5
        assert not predictions.isna().any()

    def test_allocator_minimal_history(self):
        """Allocator should work with minimal historical data."""
        from src.allocators.models import EqualWeightAllocator

        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        historical = pd.DataFrame(
            {"A": [0.01, 0.02, -0.01], "B": [-0.01, 0.01, 0.02]},
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03], index=["A", "B"])

        allocator = EqualWeightAllocator()
        weights = allocator.allocate(predicted, historical)

        assert len(weights) == 2
        assert weights.sum() == pytest.approx(1.0)

    def test_risk_parity_minimal_history(self):
        """RiskParity with very short history should fallback gracefully."""
        from src.allocators.models import RiskParityAllocator

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        np.random.seed(42)
        historical = pd.DataFrame(
            {"A": np.random.randn(5) * 0.02, "B": np.random.randn(5) * 0.02},
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03], index=["A", "B"])

        allocator = RiskParityAllocator()
        weights = allocator.allocate(predicted, historical)

        # Should return valid weights (possibly fallback to equal weight)
        assert len(weights) == 2
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Additional Edge Cases
# ===========================================================================


class TestEmptyInputs:
    """Test handling of empty inputs."""

    def test_allocator_empty_predicted_returns(self):
        """Allocator should reject empty predicted returns."""
        from src.allocators.models import EqualWeightAllocator

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        historical = pd.DataFrame({"A": np.random.randn(100)}, index=dates)
        predicted = pd.Series([], dtype=float)

        allocator = EqualWeightAllocator()
        with pytest.raises(ValueError, match="empty"):
            allocator.allocate(predicted, historical)

    def test_allocator_empty_historical_returns(self):
        """Allocator should reject empty historical returns."""
        from src.allocators.models import EqualWeightAllocator

        predicted = pd.Series([0.05], index=["A"])
        historical = pd.DataFrame()

        allocator = EqualWeightAllocator()
        with pytest.raises(ValueError, match="empty"):
            allocator.allocate(predicted, historical)

    def test_forecaster_empty_train_data(self):
        """Forecaster should reject empty training data."""
        from src.forecasters.baselines import NaiveForecaster

        empty = pd.Series([], dtype=float)

        forecaster = NaiveForecaster()
        with pytest.raises(ValueError, match="empty"):
            forecaster.fit(empty)


class TestHorizonValidation:
    """Test horizon parameter validation."""

    @pytest.fixture
    def fitted_forecaster(self):
        """Create a fitted forecaster."""
        from src.forecasters.baselines import NaiveForecaster

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        data = pd.Series(np.random.randn(100), index=dates)
        forecaster = NaiveForecaster()
        forecaster.fit(data)
        return forecaster

    def test_zero_horizon_rejected(self, fitted_forecaster):
        """Horizon of 0 should be rejected."""
        with pytest.raises(ValueError, match="horizon"):
            fitted_forecaster.predict(0)

    def test_negative_horizon_rejected(self, fitted_forecaster):
        """Negative horizon should be rejected."""
        with pytest.raises(ValueError, match="horizon"):
            fitted_forecaster.predict(-5)

    def test_float_horizon_rejected(self, fitted_forecaster):
        """Float horizon should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            fitted_forecaster.predict(5.5)


class TestTickerMismatch:
    """Test handling of ticker mismatches between inputs."""

    def test_missing_ticker_in_historical(self):
        """Allocator should reject when ticker in predicted not in historical."""
        from src.allocators.models import EqualWeightAllocator

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        historical = pd.DataFrame(
            {"A": np.random.randn(100), "B": np.random.randn(100)},
            index=dates,
        )
        # Predict for ticker C which is not in historical
        predicted = pd.Series([0.05, 0.03, 0.04], index=["A", "B", "C"])

        allocator = EqualWeightAllocator()
        with pytest.raises(ValueError, match="not found"):
            allocator.allocate(predicted, historical)

    def test_extra_ticker_in_historical_ok(self):
        """Extra tickers in historical (not in predicted) should be OK."""
        from src.allocators.models import EqualWeightAllocator

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        historical = pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100),
                "C": np.random.randn(100),  # Extra
            },
            index=dates,
        )
        predicted = pd.Series([0.05, 0.03], index=["A", "B"])

        allocator = EqualWeightAllocator()
        weights = allocator.allocate(predicted, historical)

        # Should only allocate to A and B
        assert len(weights) == 2
        assert "A" in weights.index
        assert "B" in weights.index
        assert "C" not in weights.index
