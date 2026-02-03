"""
Tests for Configuration Module
==============================

Tests configuration loading, validation, and error handling.
"""

import pytest

from src.config import (
    AllocatorsConfig,
    BacktestingConfig,
    ConfigValidationError,
    ExecutionConfig,
    ForecastersConfig,
    LoggingConfig,
    RiskConfig,
)


class TestExecutionConfigValidation:
    """Test ExecutionConfig validation."""

    def test_valid_config(self):
        """Valid execution config should work."""
        config = ExecutionConfig(mode="single", horizon=21)
        assert config.mode == "single"
        assert config.horizon == 21

    def test_invalid_mode(self):
        """Invalid mode should raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Invalid execution mode"):
            ExecutionConfig(mode="invalid_mode")

    def test_valid_modes(self):
        """All valid modes should work."""
        for mode in ExecutionConfig.VALID_MODES:
            config = ExecutionConfig(mode=mode)
            assert config.mode == mode

    def test_horizon_too_small(self):
        """Horizon < 1 should raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Horizon must be >= 1"):
            ExecutionConfig(horizon=0)

    def test_horizon_too_large(self):
        """Horizon > 365 should raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Horizon must be <= 365"):
            ExecutionConfig(horizon=400)

    def test_horizon_boundary_values(self):
        """Boundary horizon values should work."""
        config_min = ExecutionConfig(horizon=1)
        assert config_min.horizon == 1

        config_max = ExecutionConfig(horizon=365)
        assert config_max.horizon == 365


class TestRiskConfigValidation:
    """Test RiskConfig validation."""

    def test_valid_config(self):
        """Valid risk config should work."""
        config = RiskConfig(minimum_allocation=0.05, maximum_allocation=0.5, risk_aversion=5.0)
        assert config.minimum_allocation == 0.05
        assert config.maximum_allocation == 0.5
        assert config.risk_aversion == 5.0

    def test_minimum_allocation_below_zero(self):
        """minimum_allocation < 0 should raise error."""
        with pytest.raises(
            ConfigValidationError, match="minimum_allocation must be between 0 and 1"
        ):
            RiskConfig(minimum_allocation=-0.1)

    def test_minimum_allocation_above_one(self):
        """minimum_allocation > 1 should raise error."""
        with pytest.raises(
            ConfigValidationError, match="minimum_allocation must be between 0 and 1"
        ):
            RiskConfig(minimum_allocation=1.5)

    def test_maximum_allocation_below_zero(self):
        """maximum_allocation < 0 should raise error."""
        with pytest.raises(
            ConfigValidationError, match="maximum_allocation must be between 0 and 1"
        ):
            RiskConfig(maximum_allocation=-0.1)

    def test_maximum_allocation_above_one(self):
        """maximum_allocation > 1 should raise error."""
        with pytest.raises(
            ConfigValidationError, match="maximum_allocation must be between 0 and 1"
        ):
            RiskConfig(maximum_allocation=1.5)

    def test_min_exceeds_max(self):
        """minimum_allocation > maximum_allocation should raise error."""
        with pytest.raises(ConfigValidationError, match="cannot exceed"):
            RiskConfig(minimum_allocation=0.5, maximum_allocation=0.3)

    def test_risk_aversion_must_be_positive(self):
        """risk_aversion <= 0 should raise error."""
        with pytest.raises(ConfigValidationError, match="risk_aversion must be > 0"):
            RiskConfig(risk_aversion=0)

        with pytest.raises(ConfigValidationError, match="risk_aversion must be > 0"):
            RiskConfig(risk_aversion=-1)


class TestForecastersConfigValidation:
    """Test ForecastersConfig validation."""

    def test_valid_config(self):
        """Valid forecasters config should work."""
        config = ForecastersConfig(
            mode="single",
            training_mode="individual",
            selection_metric="rmse",
            selection_backtest_steps=30,
        )
        assert config.mode == "single"
        assert config.training_mode == "individual"

    def test_invalid_mode(self):
        """Invalid forecaster mode should raise error."""
        with pytest.raises(ConfigValidationError, match="Invalid forecasters mode"):
            ForecastersConfig(mode="invalid")

    def test_valid_modes(self):
        """All valid forecaster modes should work."""
        for mode in ForecastersConfig.VALID_MODES:
            config = ForecastersConfig(mode=mode)
            assert config.mode == mode

    def test_invalid_training_mode(self):
        """Invalid training mode should raise error."""
        with pytest.raises(ConfigValidationError, match="Invalid training_mode"):
            ForecastersConfig(training_mode="invalid")

    def test_invalid_selection_metric(self):
        """Invalid selection metric should raise error."""
        with pytest.raises(ConfigValidationError, match="Invalid selection_metric"):
            ForecastersConfig(selection_metric="invalid_metric")

    def test_selection_backtest_steps_too_small(self):
        """selection_backtest_steps < 1 should raise error."""
        with pytest.raises(ConfigValidationError, match="selection_backtest_steps must be >= 1"):
            ForecastersConfig(selection_backtest_steps=0)


class TestBacktestingConfigValidation:
    """Test BacktestingConfig validation."""

    def test_valid_config(self):
        """Valid backtesting config should work."""
        config = BacktestingConfig(
            initial_train_size=252, test_size=1, step_size=1, min_train_size=60
        )
        assert config.initial_train_size == 252

    def test_initial_train_size_too_small(self):
        """initial_train_size < 10 should raise error."""
        with pytest.raises(ConfigValidationError, match="initial_train_size must be >= 10"):
            BacktestingConfig(initial_train_size=5)

    def test_test_size_too_small(self):
        """test_size < 1 should raise error."""
        with pytest.raises(ConfigValidationError, match="test_size must be >= 1"):
            BacktestingConfig(test_size=0)

    def test_step_size_too_small(self):
        """step_size < 1 should raise error."""
        with pytest.raises(ConfigValidationError, match="step_size must be >= 1"):
            BacktestingConfig(step_size=0)

    def test_min_train_size_too_small(self):
        """min_train_size < 10 should raise error."""
        with pytest.raises(ConfigValidationError, match="min_train_size must be >= 10"):
            BacktestingConfig(min_train_size=5)

    def test_min_exceeds_initial(self):
        """min_train_size > initial_train_size should raise error."""
        with pytest.raises(ConfigValidationError, match="cannot exceed"):
            BacktestingConfig(initial_train_size=100, min_train_size=150)


class TestLoggingConfigValidation:
    """Test LoggingConfig validation."""

    def test_valid_config(self):
        """Valid logging config should work."""
        config = LoggingConfig(level="INFO")
        assert config.level == "INFO"

    def test_invalid_level(self):
        """Invalid logging level should raise error."""
        with pytest.raises(ConfigValidationError, match="Invalid logging level"):
            LoggingConfig(level="INVALID")

    def test_valid_levels(self):
        """All valid logging levels should work."""
        for level in LoggingConfig.VALID_LEVELS:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_case_insensitive_validation(self):
        """Logging level validation should be case-insensitive."""
        # Level is stored as-is but validation is case-insensitive
        config = LoggingConfig(level="info")
        assert config.level == "info"


class TestAllocatorsConfig:
    """Test AllocatorsConfig (no validation, just structure)."""

    def test_default_config(self):
        """Default allocators config should have expected values."""
        config = AllocatorsConfig()
        assert config.default == "robust_markowitz"
        assert config.configs == {}

    def test_custom_config(self):
        """Custom allocators config should work."""
        config = AllocatorsConfig(default="risk_parity", configs={"risk_parity": {"lookback": 120}})
        assert config.default == "risk_parity"
        assert config.configs["risk_parity"]["lookback"] == 120
