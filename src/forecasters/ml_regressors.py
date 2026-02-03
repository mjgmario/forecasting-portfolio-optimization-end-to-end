"""
Machine Learning Forecasters using Scikit-learn and XGBoost

These forecasters transform time series forecasting into supervised learning
by creating features from historical data (lags, rolling stats, etc.).

Forecasting Strategies
----------------------
There are three main strategies for multi-step forecasting:

1. **Direct Strategy** (used here when horizon=h at training time):
   - Train ONE model to predict h steps ahead directly
   - Target: y[t+h] (the value h steps in the future)
   - Pros: Simple, no error accumulation
   - Cons: Need to retrain for different horizons

2. **Recursive Strategy** (used here when horizon < predict horizon):
   - Train model to predict 1 step ahead
   - Use predictions as inputs for next step
   - Pros: One model for any horizon
   - Cons: Errors accumulate over steps

3. **Multi-Output / MIMO** (not implemented):
   - Train model to predict ALL horizons at once
   - Target: [y[t+1], y[t+2], ..., y[t+h]]
   - Pros: Captures cross-horizon dependencies
   - Cons: Complex, needs multi-output models

This module uses DIRECT strategy by default (train with target h steps ahead),
with RECURSIVE fallback for horizons > 1 (predict iteratively).

Models
------
- Ridge/Lasso/ElasticNet: Linear models with regularization
- RandomForest: Ensemble of decision trees
- XGBoost: Gradient boosting (state-of-the-art)
- LightGBM/CatBoost: Alternative boosting implementations

Advantages:
- Can capture non-linear patterns
- Feature importance interpretation
- Often outperform classical methods on financial data
- No assumption of stationarity
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from xgboost import XGBRegressor

from src.constants import DEFAULT_LAGS, DEFAULT_WINDOWS, EARLY_STOPPING_SPLIT

from .base import BaseForecaster
from .ml_features import create_features, create_time_features, prepare_ml_data
from .registry import register_forecaster

logger = logging.getLogger(__name__)


class BaseMLForecaster(BaseForecaster):
    """
    Base class for ML forecasters with shared feature engineering and prediction logic.

    This class provides:
    - Common initialization for ML hyperparameters (lags, windows, etc.)
    - Feature preparation via `_prepare_data()`
    - Feature creation for forecasting via `_create_forecast_features()`
    - Iterative multi-step prediction via `_predict_iterative()`

    Subclasses only need to implement:
    - `fit()`: Train the specific model
    - Optionally override `predict()` if custom logic is needed
    """

    family = "ml"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.lags = self.config.get("lags", DEFAULT_LAGS)
        self.windows = self.config.get("windows", DEFAULT_WINDOWS)
        self.horizon = self.config.get("horizon", 1)
        self.include_time = self.config.get("include_time", True)
        self.include_technical = self.config.get("include_technical", False)
        self.diff = self.config.get("diff", True)  # Predict returns vs prices

        self.model = None
        self.feature_names = None
        self.last_features = None

    def _prepare_data(self, train_data: pd.Series):
        """Prepare features and target for training."""
        X, y = prepare_ml_data(
            train_data,
            lags=self.lags,
            windows=self.windows,
            horizon=self.horizon,
            include_time=self.include_time,
            include_technical=self.include_technical,
            diff=self.diff,
        )

        self.feature_names = X.columns.tolist()
        return X, y

    def _create_forecast_features(self, train_data: pd.Series) -> pd.DataFrame:
        """Create features for the next time step prediction."""
        # Create features from all available data
        X = create_features(
            train_data,
            lags=self.lags,
            windows=self.windows,
            include_time=False,  # Will add later for future date
            include_technical=self.include_technical,
            diff=self.diff,
        )

        # Get the last row (most recent features)
        last_features = X.iloc[[-1]].copy()

        # Add time features for the next trading day
        if self.include_time:
            last_date = train_data.index[-1]
            freq = pd.infer_freq(train_data.index) or "B"
            next_date = pd.date_range(start=last_date, periods=2, freq=freq)[1]
            time_feats = create_time_features(pd.DatetimeIndex([next_date]))
            for col in time_feats.columns:
                last_features[col] = time_feats[col].values[0]

        # Reorder columns to match training
        last_features = last_features[self.feature_names]

        return last_features

    def _predict_iterative(self, horizon: int) -> pd.Series:
        """
        Generate multi-step forecasts using iterative (recursive) prediction.

        This is the common prediction logic for all ML forecasters:
        1. Create features from current data
        2. Predict next step
        3. Append prediction to data and repeat

        :param horizon: Number of steps to forecast
        :return: Series of predicted prices with future dates as index
        """
        predictions = []
        current_data = self._last_train_data.copy()

        for _h in range(horizon):
            # Create features for next step
            X_forecast = self._create_forecast_features(current_data)

            # Predict using the trained model
            y_pred = self.model.predict(X_forecast)[0]
            predictions.append(y_pred)

            # For iterative forecasting, append prediction to series
            last_date = current_data.index[-1]
            freq = pd.infer_freq(current_data.index) or "B"
            next_date = pd.date_range(start=last_date, periods=2, freq=freq)[1]

            if self.diff:
                # Predicted return -> convert to price
                next_price = current_data.iloc[-1] * (1 + y_pred)
            else:
                next_price = y_pred

            current_data = pd.concat([current_data, pd.Series([next_price], index=[next_date])])

        # Generate future dates
        last_date = self._last_train_data.index[-1]
        freq = pd.infer_freq(self._last_train_data.index) or "B"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        # Convert returns to prices if needed
        if self.diff:
            prices = [self._last_train_data.iloc[-1]]
            for ret in predictions:
                prices.append(prices[-1] * (1 + ret))
            predictions_prices = prices[1:]
        else:
            predictions_prices = predictions

        return pd.Series(data=predictions_prices, index=future_dates)

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Generate forecasts using iterative prediction.

        :param horizon: Number of steps to forecast
        :return: Series of predicted prices
        """
        self._validate_fitted()
        self._validate_horizon(horizon)

        if horizon != self.horizon:
            logger.warning(
                f"Model trained for horizon={self.horizon}, predicting for horizon={horizon}. "
                "Results may be suboptimal. Consider retraining."
            )

        result = self._predict_iterative(horizon)
        logger.debug(f"Generated {horizon}-step {self.name} forecast")
        return result


@register_forecaster("ridge")
class RidgeForecaster(BaseMLForecaster):
    """
    Ridge Regression forecaster (L2 regularization).

    Linear model with L2 penalty on coefficients.
    Good baseline for ML forecasting - fast, interpretable, robust.

    Use cases:
    - Linear baseline for ML models
    - When interpretability is important
    - Quick training on large datasets

    Config:
        alpha: float - Regularization strength (default: 1.0)
        lags: int - Number of lag features (default: 5)
        windows: list - Rolling window sizes (default: [5, 20])
        horizon: int - Forecast horizon (default: 1)
        include_time: bool - Include time features (default: True)
        include_technical: bool - Include RSI, MACD (default: False)
        diff: bool - Predict returns vs prices (default: True)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 1.0)

    def fit(self, train_data: pd.Series) -> "RidgeForecaster":
        """Fit Ridge regression model."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 10:
            raise ValueError("Insufficient data after feature creation. Need more training data.")

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(
            f"Fitted {self.name} on {len(X)} samples with {X.shape[1]} features "
            f"(alpha={self.alpha})"
        )
        return self

    # predict() inherited from BaseMLForecaster


@register_forecaster("lasso")
class LassoForecaster(BaseMLForecaster):
    """
    Lasso Regression forecaster (L1 regularization).

    Linear model with L1 penalty - performs feature selection by
    setting some coefficients to zero.

    Advantages over Ridge:
    - Automatic feature selection
    - Sparser models
    - Better when many features are irrelevant

    Config:
        alpha: float - Regularization strength (default: 0.01)
        lags: int - Number of lags (default: 5)
        windows: list - Rolling windows (default: [5, 20])
        (same as Ridge for other params)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 0.01)

    def fit(self, train_data: pd.Series) -> "LassoForecaster":
        """Fit Lasso regression model."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 10:
            raise ValueError("Insufficient data after feature creation.")

        self.model = Lasso(alpha=self.alpha, max_iter=10000)
        self.model.fit(X, y)

        self._last_train_data = train_data
        self.is_fitted = True

        # Log feature selection
        n_nonzero = np.sum(self.model.coef_ != 0)
        logger.debug(
            f"Fitted {self.name} on {len(X)} samples with {X.shape[1]} features "
            f"({n_nonzero} selected, alpha={self.alpha})"
        )
        return self

    # predict() inherited from BaseMLForecaster


@register_forecaster("random_forest")
class RandomForestForecaster(BaseMLForecaster):
    """
    Random Forest forecaster with advanced hyperparameters.

    Ensemble of decision trees - can capture non-linear patterns.
    Uses bagging (Bootstrap Aggregating) for variance reduction.

    Advantages:
    - Handles non-linearity and interactions
    - Feature importance via MDI/permutation
    - Robust to outliers and noise
    - No feature scaling needed
    - Out-of-bag (OOB) error for validation

    Config:
        n_estimators: int - Number of trees (default: 200)
        max_depth: int - Max tree depth, None for unlimited (default: 15)
        min_samples_split: int - Min samples to split internal node (default: 5)
        min_samples_leaf: int - Min samples at leaf node (default: 2)
        max_features: str/float - Features per split: "sqrt", "log2", or fraction (default: 0.8)
        max_samples: float - Bootstrap sample size as fraction (default: 0.8)
        bootstrap: bool - Use bootstrap sampling (default: True)
        oob_score: bool - Compute out-of-bag score (default: True)
        criterion: str - Split criterion: "squared_error", "absolute_error" (default: "squared_error")
        ccp_alpha: float - Cost complexity pruning (default: 0.0)
        lags: int - Number of lag features (default: 5)
        windows: list - Rolling window sizes (default: [5, 20])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", 15)
        self.min_samples_split = self.config.get("min_samples_split", 5)
        self.min_samples_leaf = self.config.get("min_samples_leaf", 2)
        self.max_features = self.config.get("max_features", 0.8)
        self.max_samples = self.config.get("max_samples", 0.8)
        self.bootstrap = self.config.get("bootstrap", True)
        self.oob_score = self.config.get("oob_score", True)
        self.criterion = self.config.get("criterion", "squared_error")
        self.ccp_alpha = self.config.get("ccp_alpha", 0.0)

    def fit(self, train_data: pd.Series) -> "RandomForestForecaster":
        """Fit Random Forest model with advanced parameters."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 20:
            raise ValueError("Random Forest requires more training data (>20 samples).")

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_samples=self.max_samples if self.bootstrap else None,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score and self.bootstrap,
            criterion=self.criterion,
            ccp_alpha=self.ccp_alpha,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        self._last_train_data = train_data
        self.is_fitted = True

        oob_info = ""
        if self.oob_score and self.bootstrap and hasattr(self.model, "oob_score_"):
            oob_info = f", OOB R²={self.model.oob_score_:.4f}"

        logger.debug(
            f"Fitted {self.name} on {len(X)} samples with {X.shape[1]} features "
            f"(n_trees={self.n_estimators}, max_depth={self.max_depth}{oob_info})"
        )
        return self

    # predict() inherited from BaseMLForecaster

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances from trained model."""
        self._validate_fitted()
        return pd.Series(
            data=self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


@register_forecaster("xgboost")
class XGBoostForecaster(BaseMLForecaster):
    """
    XGBoost forecaster with advanced hyperparameters.

    Gradient boosting - state-of-the-art for tabular data.
    Uses second-order gradients for more accurate optimization.

    Advantages:
    - Best performance on structured data
    - Fast training via histogram-based splits
    - Built-in L1/L2 regularization
    - Handles missing values natively
    - Early stopping to prevent overfitting

    Config:
        n_estimators: int - Number of boosting rounds (default: 200)
        max_depth: int - Max tree depth (default: 6)
        learning_rate: float - Shrinkage/eta (default: 0.05)
        subsample: float - Row sampling per tree (default: 0.8)
        colsample_bytree: float - Feature sampling per tree (default: 0.8)
        colsample_bylevel: float - Feature sampling per level (default: 0.8)
        gamma: float - Min loss reduction for split (default: 0.1)
        min_child_weight: float - Min sum of instance weight in child (default: 3)
        reg_alpha: float - L1 regularization (default: 0.1)
        reg_lambda: float - L2 regularization (default: 1.0)
        max_delta_step: float - Max delta step for weight estimation (default: 0)
        scale_pos_weight: float - Balance for imbalanced classes (default: 1)
        tree_method: str - Tree construction algorithm (default: "hist")
        booster: str - Booster type: "gbtree", "gblinear", "dart" (default: "gbtree")
        early_stopping_rounds: int - Early stopping patience (default: 20)
        eval_metric: str - Evaluation metric (default: "rmse")
        lags: int - Number of lag features (default: 5)
        windows: list - Rolling window sizes (default: [5, 20])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", 6)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.subsample = self.config.get("subsample", 0.8)
        self.colsample_bytree = self.config.get("colsample_bytree", 0.8)
        self.colsample_bylevel = self.config.get("colsample_bylevel", 0.8)
        self.gamma = self.config.get("gamma", 0.1)
        self.min_child_weight = self.config.get("min_child_weight", 3)
        self.reg_alpha = self.config.get("reg_alpha", 0.1)
        self.reg_lambda = self.config.get("reg_lambda", 1.0)
        self.max_delta_step = self.config.get("max_delta_step", 0)
        self.tree_method = self.config.get("tree_method", "hist")
        self.booster = self.config.get("booster", "gbtree")
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 20)
        self.eval_metric = self.config.get("eval_metric", "rmse")

    def fit(self, train_data: pd.Series) -> "XGBoostForecaster":
        """Fit XGBoost model with advanced parameters and optional early stopping."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 20:
            raise ValueError("XGBoost requires more training data (>20 samples).")

        # Split for early stopping validation if enough data
        use_early_stopping = len(X) > 50 and self.early_stopping_rounds > 0
        eval_set = None

        if use_early_stopping:
            split_idx = int(len(X) * EARLY_STOPPING_SPLIT)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            max_delta_step=self.max_delta_step,
            tree_method=self.tree_method,
            booster=self.booster,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds if use_early_stopping else None,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self._last_train_data = train_data
        self.is_fitted = True

        best_iter_info = ""
        if use_early_stopping and hasattr(self.model, "best_iteration"):
            best_iter_info = f", best_iter={self.model.best_iteration}"

        logger.debug(
            f"Fitted {self.name} on {len(X_train)} samples with {X.shape[1]} features "
            f"(n_estimators={self.n_estimators}, lr={self.learning_rate}{best_iter_info})"
        )
        return self

    # predict() inherited from BaseMLForecaster

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances from trained XGBoost model."""
        self._validate_fitted()
        return pd.Series(
            data=self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


@register_forecaster("elasticnet")
class ElasticNetForecaster(BaseMLForecaster):
    """
    ElasticNet Regression forecaster (L1 + L2 regularization).

    Combines Ridge (L2) and Lasso (L1) penalties.
    Good for feature selection with stability.

    ElasticNet penalty: α * (λ₁ * ||w||₁ + λ₂ * ||w||₂²)

    Use cases:
    - When features are correlated (better than Lasso alone)
    - Feature selection with stability
    - Linear baseline with regularization

    Config:
        alpha: float - Overall regularization strength (default: 1.0)
        l1_ratio: float - Balance between L1/L2 (0=Ridge, 1=Lasso) (default: 0.5)
        lags: int - Number of lags (default: 5)
        windows: list - Rolling windows (default: [5, 20])

    Example:
        >>> forecaster = ForecasterRegistry.create("elasticnet", {
        ...     "alpha": 0.1,
        ...     "l1_ratio": 0.5
        ... })
        >>> forecaster.fit(price_series)
        >>> predictions = forecaster.predict(horizon=5)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 1.0)
        self.l1_ratio = self.config.get("l1_ratio", 0.5)

    def fit(self, train_data: pd.Series) -> "ElasticNetForecaster":
        """Fit ElasticNet regression model."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 10:
            raise ValueError("Insufficient data after feature creation.")

        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10000)
        self.model.fit(X, y)

        self._last_train_data = train_data
        self.is_fitted = True

        n_nonzero = np.sum(self.model.coef_ != 0)
        logger.debug(
            f"Fitted {self.name} on {len(X)} samples with {X.shape[1]} features "
            f"({n_nonzero} selected, alpha={self.alpha}, l1_ratio={self.l1_ratio})"
        )
        return self

    # predict() inherited from BaseMLForecaster


@register_forecaster("hist_gradient_boosting")
class HistGradientBoostingForecaster(BaseMLForecaster):
    """
    Histogram-based Gradient Boosting forecaster (sklearn).

    Fast gradient boosting implementation inspired by LightGBM.
    Native support for missing values and categorical features.

    Advantages:
    - Much faster than GradientBoostingRegressor
    - Handles missing values natively
    - Built-in categorical support
    - Good performance with default parameters

    Use cases:
    - When you want boosting speed without XGBoost/LightGBM
    - Large datasets (>10k samples)
    - When features have missing values

    Config:
        max_iter: int - Number of boosting iterations (default: 100)
        max_depth: int - Max tree depth (default: 10)
        learning_rate: float - Shrinkage (default: 0.1)
        lags: int - Number of lags (default: 5)
        windows: list - Rolling windows (default: [5, 20])

    Example:
        >>> forecaster = ForecasterRegistry.create("hist_gradient_boosting", {
        ...     "max_iter": 200,
        ...     "learning_rate": 0.05
        ... })
        >>> forecaster.fit(price_series)
        >>> predictions = forecaster.predict(horizon=5)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_iter = self.config.get("max_iter", 100)
        self.max_depth = self.config.get("max_depth", 10)
        self.learning_rate = self.config.get("learning_rate", 0.1)

    def fit(self, train_data: pd.Series) -> "HistGradientBoostingForecaster":
        """Fit HistGradientBoosting model."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 20:
            raise ValueError("HistGradientBoosting requires more training data (>20 samples).")

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self.model.fit(X, y)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(
            f"Fitted {self.name} on {len(X)} samples with {X.shape[1]} features "
            f"(max_iter={self.max_iter}, lr={self.learning_rate})"
        )
        return self

    # predict() inherited from BaseMLForecaster


@register_forecaster("lightgbm")
class LightGBMForecaster(BaseMLForecaster):
    """
    LightGBM forecaster with advanced hyperparameters.

    Gradient boosting framework by Microsoft using leaf-wise tree growth.
    Faster and often more accurate than XGBoost.

    Advantages:
    - Very fast training (histogram-based)
    - Low memory usage
    - Handles categorical features natively
    - Leaf-wise growth for better accuracy
    - Built-in early stopping

    Config:
        n_estimators: int - Number of boosting rounds (default: 200)
        max_depth: int - Max tree depth, -1 for no limit (default: -1)
        learning_rate: float - Shrinkage (default: 0.05)
        num_leaves: int - Max leaves per tree (default: 63)
        min_child_samples: int - Min data in leaf (default: 20)
        min_child_weight: float - Min sum of instance weight in leaf (default: 1e-3)
        subsample: float - Row sampling (default: 0.8)
        subsample_freq: int - Frequency of subsampling (default: 1)
        colsample_bytree: float - Feature sampling (default: 0.8)
        reg_alpha: float - L1 regularization (default: 0.1)
        reg_lambda: float - L2 regularization (default: 0.1)
        min_split_gain: float - Min gain for split (default: 0.01)
        max_bin: int - Max bins for histogram (default: 255)
        boosting_type: str - "gbdt", "dart", "rf" (default: "gbdt")
        early_stopping_rounds: int - Early stopping patience (default: 20)
        lags: int - Number of lags (default: 5)
        windows: list - Rolling windows (default: [5, 20])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", -1)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.num_leaves = self.config.get("num_leaves", 63)
        self.min_child_samples = self.config.get("min_child_samples", 20)
        self.min_child_weight = self.config.get("min_child_weight", 1e-3)
        self.subsample = self.config.get("subsample", 0.8)
        self.subsample_freq = self.config.get("subsample_freq", 1)
        self.colsample_bytree = self.config.get("colsample_bytree", 0.8)
        self.reg_alpha = self.config.get("reg_alpha", 0.1)
        self.reg_lambda = self.config.get("reg_lambda", 0.1)
        self.min_split_gain = self.config.get("min_split_gain", 0.01)
        self.max_bin = self.config.get("max_bin", 255)
        self.boosting_type = self.config.get("boosting_type", "gbdt")
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 20)

    def fit(self, train_data: pd.Series) -> "LightGBMForecaster":
        """Fit LightGBM model with advanced parameters."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 20:
            raise ValueError("LightGBM requires more training data (>20 samples).")

        # Split for early stopping if enough data
        use_early_stopping = len(X) > 50 and self.early_stopping_rounds > 0

        if use_early_stopping:
            split_idx = int(len(X) * EARLY_STOPPING_SPLIT)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            eval_set = None

        self.model = LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_split_gain=self.min_split_gain,
            max_bin=self.max_bin,
            boosting_type=self.boosting_type,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        fit_params = {}
        if use_early_stopping:
            fit_params["eval_set"] = eval_set

        self.model.fit(X_train, y_train, **fit_params)

        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(
            f"Fitted {self.name} on {len(X_train)} samples with {X.shape[1]} features "
            f"(n_estimators={self.n_estimators}, lr={self.learning_rate}, leaves={self.num_leaves})"
        )
        return self

    # predict() inherited from BaseMLForecaster

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances from trained LightGBM model."""
        self._validate_fitted()
        return pd.Series(
            data=self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


@register_forecaster("catboost")
class CatBoostForecaster(BaseMLForecaster):
    """
    CatBoost forecaster with advanced hyperparameters.

    Gradient boosting by Yandex using symmetric tree structure.
    Often achieves best results with minimal tuning.

    Advantages:
    - Excellent default parameters
    - Handles categorical features natively
    - Built-in overfitting detection
    - Symmetric tree structure (faster inference)
    - Ordered boosting for better generalization

    Config:
        iterations: int - Number of boosting rounds (default: 200)
        depth: int - Tree depth (default: 8)
        learning_rate: float - Shrinkage (default: 0.05)
        l2_leaf_reg: float - L2 regularization (default: 3.0)
        random_strength: float - Random forest strength (default: 1.0)
        bagging_temperature: float - Bayesian bootstrap (default: 1.0)
        border_count: int - Number of splits (default: 254)
        grow_policy: str - Tree growth policy: "SymmetricTree", "Depthwise", "Lossguide" (default: "SymmetricTree")
        min_data_in_leaf: int - Min samples in leaf (default: 1)
        max_leaves: int - Max leaves for Lossguide (default: 31)
        subsample: float - Row sampling (default: 0.8)
        rsm: float - Feature sampling (default: 0.8)
        boosting_type: str - "Ordered" or "Plain" (default: "Ordered")
        early_stopping_rounds: int - Early stopping patience (default: 20)
        lags: int - Number of lags (default: 5)
        windows: list - Rolling windows (default: [5, 20])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.iterations = self.config.get("iterations", 200)
        self.depth = self.config.get("depth", 8)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.l2_leaf_reg = self.config.get("l2_leaf_reg", 3.0)
        self.random_strength = self.config.get("random_strength", 1.0)
        self.bagging_temperature = self.config.get("bagging_temperature", 1.0)
        self.border_count = self.config.get("border_count", 254)
        self.grow_policy = self.config.get("grow_policy", "SymmetricTree")
        self.min_data_in_leaf = self.config.get("min_data_in_leaf", 1)
        self.max_leaves = self.config.get("max_leaves", 31)
        self.subsample = self.config.get("subsample", 0.8)
        self.rsm = self.config.get("rsm", 0.8)
        self.boosting_type = self.config.get("boosting_type", "Ordered")
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 20)

    def fit(self, train_data: pd.Series) -> "CatBoostForecaster":
        """Fit CatBoost model with advanced parameters."""
        self._validate_train_data(train_data)

        X, y = self._prepare_data(train_data)

        if len(X) < 20:
            raise ValueError("CatBoost requires more training data (>20 samples).")

        # Split for early stopping if enough data
        use_early_stopping = len(X) > 50 and self.early_stopping_rounds > 0

        if use_early_stopping:
            split_idx = int(len(X) * EARLY_STOPPING_SPLIT)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X, y
            eval_set = None

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            border_count=self.border_count,
            grow_policy=self.grow_policy,
            min_data_in_leaf=self.min_data_in_leaf,
            max_leaves=self.max_leaves if self.grow_policy == "Lossguide" else None,
            subsample=self.subsample,
            rsm=self.rsm,
            boosting_type=self.boosting_type,
            random_state=42,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds if use_early_stopping else None,
        )

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        self._last_train_data = train_data
        self.is_fitted = True

        best_iter_info = ""
        if use_early_stopping and hasattr(self.model, "best_iteration_"):
            best_iter_info = f", best_iter={self.model.best_iteration_}"

        logger.debug(
            f"Fitted {self.name} on {len(X_train)} samples with {X.shape[1]} features "
            f"(iterations={self.iterations}, lr={self.learning_rate}{best_iter_info})"
        )
        return self

    # predict() inherited from BaseMLForecaster

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances from trained CatBoost model."""
        self._validate_fitted()
        return pd.Series(
            data=self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


__all__ = [
    "RidgeForecaster",
    "LassoForecaster",
    "ElasticNetForecaster",
    "RandomForestForecaster",
    "HistGradientBoostingForecaster",
    "XGBoostForecaster",
    "LightGBMForecaster",
    "CatBoostForecaster",
]
