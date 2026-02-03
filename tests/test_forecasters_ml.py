"""Tests for machine learning forecasters.

This module tests:
- RidgeForecaster
- LassoForecaster
- ElasticNetForecaster
- RandomForestForecaster
- HistGradientBoostingForecaster
- XGBoostForecaster
- LightGBMForecaster
- CatBoostForecaster
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasters import ForecasterRegistry
from src.forecasters.ml_regressors import (
    CatBoostForecaster,
    ElasticNetForecaster,
    HistGradientBoostingForecaster,
    LassoForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)


@pytest.fixture
def sample_price_series():
    """Create a sample price time series for ML models."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    # Random walk with trend and noise
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    prices = trend + np.cumsum(noise) * 0.1
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_return_series():
    """Create a sample return time series for ML models."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    returns = np.random.randn(200) * 0.02  # 2% volatility
    return pd.Series(returns, index=dates)


class TestRidgeForecaster:
    """Test Ridge regression forecaster."""

    def test_registration(self):
        """Test that Ridge forecaster is registered."""
        assert "ridge" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("ridge")
        assert forecaster.__class__.__name__ == "RidgeForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting Ridge model."""
        forecaster = RidgeForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = RidgeForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(~predictions.isna())

    def test_with_custom_alpha(self, sample_price_series):
        """Test with custom regularization parameter."""
        forecaster = RidgeForecaster(config={"alpha": 10.0, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.alpha == 10.0

    def test_family(self):
        """Test that Ridge has family='ml'."""
        forecaster = RidgeForecaster()
        assert forecaster.family == "ml"


class TestLassoForecaster:
    """Test Lasso regression forecaster."""

    def test_registration(self):
        """Test that Lasso forecaster is registered."""
        assert "lasso" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("lasso")
        assert forecaster.__class__.__name__ == "LassoForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting Lasso model."""
        forecaster = LassoForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = LassoForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5

    def test_with_custom_alpha(self, sample_price_series):
        """Test with custom regularization parameter."""
        forecaster = LassoForecaster(config={"alpha": 0.01, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.alpha == 0.01


class TestElasticNetForecaster:
    """Test ElasticNet regression forecaster."""

    def test_registration(self):
        """Test that ElasticNet forecaster is registered."""
        assert "elasticnet" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("elasticnet")
        assert forecaster.__class__.__name__ == "ElasticNetForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting ElasticNet model."""
        forecaster = ElasticNetForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = ElasticNetForecaster(config={"lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(~predictions.isna())

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom alpha and l1_ratio."""
        forecaster = ElasticNetForecaster(config={"alpha": 0.5, "l1_ratio": 0.7, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.alpha == 0.5
        assert forecaster.l1_ratio == 0.7

    def test_l1_ratio_extremes(self, sample_price_series):
        """Test l1_ratio at extremes (pure Lasso and pure Ridge)."""
        # l1_ratio=1.0 should be pure Lasso
        forecaster_lasso = ElasticNetForecaster(config={"alpha": 1.0, "l1_ratio": 1.0, "lags": 5})
        forecaster_lasso.fit(sample_price_series)

        # l1_ratio=0.0 should be pure Ridge
        forecaster_ridge = ElasticNetForecaster(config={"alpha": 1.0, "l1_ratio": 0.0, "lags": 5})
        forecaster_ridge.fit(sample_price_series)

        assert forecaster_lasso.is_fitted
        assert forecaster_ridge.is_fitted

    def test_family(self):
        """Test that ElasticNet has family='ml'."""
        forecaster = ElasticNetForecaster()
        assert forecaster.family == "ml"


class TestRandomForestForecaster:
    """Test Random Forest forecaster."""

    def test_registration(self):
        """Test that Random Forest forecaster is registered."""
        assert "random_forest" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("random_forest")
        assert forecaster.__class__.__name__ == "RandomForestForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting Random Forest model."""
        forecaster = RandomForestForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = RandomForestForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom hyperparameters."""
        config = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "lags": 5,
        }
        forecaster = RandomForestForecaster(config=config)
        forecaster.fit(sample_price_series)

        assert forecaster.n_estimators == 100
        assert forecaster.max_depth == 10


class TestHistGradientBoostingForecaster:
    """Test Histogram-based Gradient Boosting forecaster."""

    def test_registration(self):
        """Test that HistGradientBoosting forecaster is registered."""
        assert "hist_gradient_boosting" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("hist_gradient_boosting")
        assert forecaster.__class__.__name__ == "HistGradientBoostingForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting HistGradientBoosting model."""
        forecaster = HistGradientBoostingForecaster(config={"max_iter": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = HistGradientBoostingForecaster(config={"max_iter": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(~predictions.isna())

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom hyperparameters."""
        config = {
            "max_iter": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "lags": 10,
        }
        forecaster = HistGradientBoostingForecaster(config=config)
        forecaster.fit(sample_price_series)

        assert forecaster.max_iter == 200
        assert forecaster.learning_rate == 0.05

    def test_family(self):
        """Test that HistGradientBoosting has family='ml'."""
        forecaster = HistGradientBoostingForecaster()
        assert forecaster.family == "ml"


class TestXGBoostForecaster:
    """Test XGBoost forecaster."""

    def test_registration(self):
        """Test that XGBoost forecaster is registered."""
        assert "xgboost" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("xgboost")
        assert forecaster.__class__.__name__ == "XGBoostForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting XGBoost model."""
        forecaster = XGBoostForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = XGBoostForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom hyperparameters."""
        config = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "lags": 10,
        }
        forecaster = XGBoostForecaster(config=config)
        forecaster.fit(sample_price_series)

        assert forecaster.n_estimators == 200
        assert forecaster.learning_rate == 0.05


class TestLightGBMForecaster:
    """Test LightGBM forecaster."""

    def test_registration(self):
        """Test that LightGBM forecaster is registered."""
        assert "lightgbm" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("lightgbm")
        assert forecaster.__class__.__name__ == "LightGBMForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting LightGBM model."""
        forecaster = LightGBMForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = LightGBMForecaster(config={"n_estimators": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(~predictions.isna())

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom hyperparameters."""
        config = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "lags": 10,
        }
        forecaster = LightGBMForecaster(config=config)
        forecaster.fit(sample_price_series)

        assert forecaster.n_estimators == 100
        assert forecaster.learning_rate == 0.05

    def test_family(self):
        """Test that LightGBM has family='ml'."""
        forecaster = LightGBMForecaster()
        assert forecaster.family == "ml"


class TestCatBoostForecaster:
    """Test CatBoost forecaster."""

    def test_registration(self):
        """Test that CatBoost forecaster is registered."""
        assert "catboost" in ForecasterRegistry.list_all()
        forecaster = ForecasterRegistry.create("catboost")
        assert forecaster.__class__.__name__ == "CatBoostForecaster"

    def test_fit(self, sample_price_series):
        """Test fitting CatBoost model."""
        forecaster = CatBoostForecaster(config={"iterations": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        assert forecaster.is_fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_series):
        """Test prediction."""
        forecaster = CatBoostForecaster(config={"iterations": 50, "lags": 5})
        forecaster.fit(sample_price_series)

        predictions = forecaster.predict(horizon=5)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(~predictions.isna())

    def test_with_custom_hyperparameters(self, sample_price_series):
        """Test with custom hyperparameters."""
        config = {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.05,
            "lags": 10,
        }
        forecaster = CatBoostForecaster(config=config)
        forecaster.fit(sample_price_series)

        assert forecaster.iterations == 100
        assert forecaster.learning_rate == 0.05

    def test_family(self):
        """Test that CatBoost has family='ml'."""
        forecaster = CatBoostForecaster()
        assert forecaster.family == "ml"


class TestMLForecasterComparison:
    """Test comparison of ML forecasters."""

    def test_all_ml_models_work_on_same_data(self, sample_price_series):
        """Test that all ML models can fit and predict on same data."""
        ml_models = [
            "ridge",
            "lasso",
            "elasticnet",
            "random_forest",
            "hist_gradient_boosting",
        ]
        ml_models.append("xgboost")

        for model_name in ml_models:
            forecaster = ForecasterRegistry.create(
                model_name, config={"lags": 5, "n_estimators": 10}
            )
            forecaster.fit(sample_price_series)
            predictions = forecaster.predict(horizon=3)

            assert isinstance(predictions, pd.Series)
            assert len(predictions) == 3
            assert all(~predictions.isna())

    def test_family_attribute(self):
        """Test that all ML forecasters have family='ml'."""
        ml_models = [
            "ridge",
            "lasso",
            "elasticnet",
            "random_forest",
            "hist_gradient_boosting",
        ]
        ml_models.append("xgboost")

        for model_name in ml_models:
            forecaster = ForecasterRegistry.create(model_name)
            assert forecaster.family == "ml"

    def test_feature_engineering_consistency(self, sample_price_series):
        """Test that feature engineering is consistent across models."""
        # All ML models should handle the same lags and features
        config = {"lags": 10, "windows": [5, 20], "include_technical": True}

        models_to_test = ["ridge", "xgboost"]

        for model_name in models_to_test:
            forecaster = ForecasterRegistry.create(model_name, config=config)
            forecaster.fit(sample_price_series)
            predictions = forecaster.predict(horizon=1)

            assert len(predictions) == 1
            assert ~predictions.isna().any()
