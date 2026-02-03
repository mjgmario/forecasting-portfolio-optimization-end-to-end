"""
Deep Learning Forecasters - N-BEATS and N-HiTS

State-of-the-art neural network models for time series forecasting.

Models:
- N-BEATS: Neural Basis Expansion Analysis for Time Series
- N-HiTS: Neural Hierarchical Interpolation for Time Series

These models require PyTorch and the neuralforecast library.

References:
- N-BEATS: Oreshkin et al. (2019) - https://arxiv.org/abs/1905.10437
- N-HiTS: Challu et al. (2022) - https://arxiv.org/abs/2201.12886
"""

import logging
from typing import Any

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

from .base import BaseForecaster
from .registry import register_forecaster

logger = logging.getLogger(__name__)


@register_forecaster("nbeats")
class NBeatsForecaster(BaseForecaster):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    Pure deep learning approach without hand-crafted features.
    Uses stacks of fully connected layers with residual connections.

    Architecture:
    - Stacks: Multiple blocks of FC layers
    - Each block: Basis expansion for trend and seasonality
    - Residual connections for better gradient flow

    Config:
        input_size: History length (default: 5*horizon)
        h: Forecast horizon (default: 5)
        max_steps: Training epochs (default: 500)
        learning_rate: Learning rate (default: 0.001)
        stack_types: Stack types (default: ["identity", "trend", "seasonality"])
        n_blocks: Blocks per stack (default: [1, 1, 1])
        mlp_units: Hidden units per block (default: [[512, 512], [512, 512], [512, 512]])
        batch_size: Batch size (default: 32)
        windows_batch_size: Windows per batch (default: 128)
        scaler_type: Normalization (default: "robust")
    """

    family = "deep_learning"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.h = self.config.get("h", 5)
        self.input_size = self.config.get("input_size", 5 * self.h)
        self.max_steps = self.config.get("max_steps", 500)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.stack_types = self.config.get("stack_types", ["identity", "trend", "seasonality"])
        self.n_blocks = self.config.get("n_blocks", [1, 1, 1])
        self.mlp_units = self.config.get("mlp_units", [[512, 512], [512, 512], [512, 512]])
        self.batch_size = self.config.get("batch_size", 32)
        self.windows_batch_size = self.config.get("windows_batch_size", 128)
        self.scaler_type = self.config.get("scaler_type", "robust")

        self.nf = None
        self._train_df = None

    def fit(self, train_data: pd.Series) -> "NBeatsForecaster":
        """Fit N-BEATS model."""
        self._validate_train_data(train_data)

        if len(train_data) < self.input_size + self.h:
            raise ValueError(
                f"Need at least {self.input_size + self.h} samples, got {len(train_data)}"
            )

        # Prepare data in neuralforecast format
        df = pd.DataFrame(
            {
                "unique_id": "asset",
                "ds": train_data.index,
                "y": train_data.values,
            }
        )

        # Create N-BEATS model
        models = [
            NBEATS(
                h=self.h,
                input_size=self.input_size,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                stack_types=self.stack_types,
                n_blocks=self.n_blocks,
                mlp_units=self.mlp_units,
                batch_size=self.batch_size,
                windows_batch_size=self.windows_batch_size,
                scaler_type=self.scaler_type,
                random_seed=42,
            )
        ]

        self.nf = NeuralForecast(models=models, freq="D")
        self.nf.fit(df=df)

        self._train_df = df
        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(f"N-BEATS fitted: {len(train_data)} samples, horizon={self.h}")
        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """Generate N-BEATS forecasts."""
        self._validate_fitted()
        self._validate_horizon(horizon)

        if horizon != self.h:
            logger.warning(
                f"Model trained for h={self.h}, predicting h={horizon}. "
                "Consider retraining for optimal results."
            )

        # Predict
        forecast_df = self.nf.predict()

        # Extract predictions
        predictions = forecast_df["NBEATS"].values[:horizon]

        # Generate future dates using base class helper
        last_date = self._last_train_data.index[-1]
        future_dates = self._make_future_index(last_date, horizon)

        result = pd.Series(data=predictions, index=future_dates)
        logger.debug(f"N-BEATS forecast generated: {horizon} steps")
        return result


@register_forecaster("nhits")
class NHiTSForecaster(BaseForecaster):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series.

    Improved version of N-BEATS with hierarchical interpolation.
    More efficient and often more accurate than N-BEATS.

    Architecture:
    - Multi-rate signal processing
    - Hierarchical interpolation
    - Pooling for different frequencies

    Advantages over N-BEATS:
    - Faster training (fewer parameters)
    - Better long-horizon forecasts
    - Captures multiple seasonalities better

    Config:
        input_size: History length (default: 5*horizon)
        h: Forecast horizon (default: 5)
        max_steps: Training epochs (default: 500)
        learning_rate: Learning rate (default: 0.001)
        n_pool_kernel_size: Pooling sizes (default: [2, 2, 1])
        n_freq_downsample: Downsampling rates (default: [4, 2, 1])
        interpolation_mode: Interpolation (default: "linear")
        batch_size: Batch size (default: 32)
        windows_batch_size: Windows per batch (default: 128)
        scaler_type: Normalization (default: "robust")
    """

    family = "deep_learning"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.h = self.config.get("h", 5)
        self.input_size = self.config.get("input_size", 5 * self.h)
        self.max_steps = self.config.get("max_steps", 500)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.n_pool_kernel_size = self.config.get("n_pool_kernel_size", [2, 2, 1])
        self.n_freq_downsample = self.config.get("n_freq_downsample", [4, 2, 1])
        self.interpolation_mode = self.config.get("interpolation_mode", "linear")
        self.batch_size = self.config.get("batch_size", 32)
        self.windows_batch_size = self.config.get("windows_batch_size", 128)
        self.scaler_type = self.config.get("scaler_type", "robust")

        self.nf = None
        self._train_df = None

    def fit(self, train_data: pd.Series) -> "NHiTSForecaster":
        """Fit N-HiTS model."""
        self._validate_train_data(train_data)

        if len(train_data) < self.input_size + self.h:
            raise ValueError(
                f"Need at least {self.input_size + self.h} samples, got {len(train_data)}"
            )

        # Prepare data
        df = pd.DataFrame(
            {
                "unique_id": "asset",
                "ds": train_data.index,
                "y": train_data.values,
            }
        )

        # Create N-HiTS model
        models = [
            NHITS(
                h=self.h,
                input_size=self.input_size,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                n_pool_kernel_size=self.n_pool_kernel_size,
                n_freq_downsample=self.n_freq_downsample,
                interpolation_mode=self.interpolation_mode,
                batch_size=self.batch_size,
                windows_batch_size=self.windows_batch_size,
                scaler_type=self.scaler_type,
                random_seed=42,
            )
        ]

        self.nf = NeuralForecast(models=models, freq="D")
        self.nf.fit(df=df)

        self._train_df = df
        self._last_train_data = train_data
        self.is_fitted = True

        logger.debug(f"N-HiTS fitted: {len(train_data)} samples, horizon={self.h}")
        return self

    def predict(self, horizon: int = 1) -> pd.Series:
        """Generate N-HiTS forecasts."""
        self._validate_fitted()
        self._validate_horizon(horizon)

        if horizon != self.h:
            logger.warning(
                f"Model trained for h={self.h}, predicting h={horizon}. "
                "Consider retraining for optimal results."
            )

        # Predict
        forecast_df = self.nf.predict()

        # Extract predictions
        predictions = forecast_df["NHITS"].values[:horizon]

        # Generate future dates using base class helper
        last_date = self._last_train_data.index[-1]
        future_dates = self._make_future_index(last_date, horizon)

        result = pd.Series(data=predictions, index=future_dates)
        logger.debug(f"N-HiTS forecast generated: {horizon} steps")
        return result


__all__ = ["NBeatsForecaster", "NHiTSForecaster"]
