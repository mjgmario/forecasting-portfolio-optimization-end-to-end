"""
Configuration Management Module
================================

Loads configuration from YAML files with the following priority:
1. config/local.yaml (gitignored, for personal overrides)
2. config/default.yaml (version controlled defaults)
3. Environment variables (for secrets like DATABASE_URL)

Usage:
    >>> from src.config import config
    >>> tickers = config.portfolio.tickers
    >>> forecaster_config = config.get_forecaster_config("xgboost")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    :param base: Base dictionary
    :param override: Override dictionary (takes precedence)
    :return: Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


@dataclass
class ExecutionConfig:
    """
    Execution settings configuration.

    :ivar mode: Run mode - "single", "compare", or "comprehensive"
    :ivar save_to_db: Whether to save results to database
    :ivar horizon: Forecast horizon in trading days
    :ivar verbose: Enable verbose output
    """

    mode: str = "single"  # "single", "compare", "comprehensive"
    save_to_db: bool = True
    horizon: int = 21
    verbose: bool = True

    VALID_MODES = ("single", "compare", "comprehensive")

    def __post_init__(self):
        if self.mode not in self.VALID_MODES:
            raise ConfigValidationError(
                f"Invalid execution mode '{self.mode}'. Valid modes: {self.VALID_MODES}"
            )
        if self.horizon < 1:
            raise ConfigValidationError(f"Horizon must be >= 1, got {self.horizon}")
        if self.horizon > 365:
            raise ConfigValidationError(f"Horizon must be <= 365, got {self.horizon}")


@dataclass
class RiskConfig:
    """Risk parameters configuration."""

    minimum_allocation: float = 0.05
    maximum_allocation: float = 1.0
    risk_aversion: float = 5.0

    def __post_init__(self):
        if not 0 <= self.minimum_allocation <= 1:
            raise ConfigValidationError(
                f"minimum_allocation must be between 0 and 1, got {self.minimum_allocation}"
            )
        if not 0 <= self.maximum_allocation <= 1:
            raise ConfigValidationError(
                f"maximum_allocation must be between 0 and 1, got {self.maximum_allocation}"
            )
        if self.minimum_allocation > self.maximum_allocation:
            raise ConfigValidationError(
                f"minimum_allocation ({self.minimum_allocation}) cannot exceed "
                f"maximum_allocation ({self.maximum_allocation})"
            )
        if self.risk_aversion <= 0:
            raise ConfigValidationError(f"risk_aversion must be > 0, got {self.risk_aversion}")


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""

    tickers: list[str] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)


@dataclass
class DatesConfig:
    """Date range configuration."""

    start_date: str = "2024-01-01"
    end_date: str | None = None

    @property
    def effective_end_date(self) -> str:
        """Get end date, defaulting to today if not set."""
        if self.end_date is None:
            return datetime.now().strftime("%Y-%m-%d")
        return self.end_date


@dataclass
class ForecastersConfig:
    """
    Forecasters configuration.

    :ivar default: Default forecaster when mode is "single"
    :ivar mode: Selection mode - "single", "auto", or "ensemble"
    :ivar training_mode: "individual" (per ticker) or "combined" (all tickers)
    :ivar per_ticker: Dict mapping ticker to specific forecaster
    :ivar candidates: Forecasters to evaluate in "auto" mode
    :ivar selection_metric: Metric for auto-selection
    :ivar selection_backtest_steps: Steps for auto-selection backtest
    :ivar skip: Forecasters to skip in comprehensive analysis
    :ivar configs: Per-forecaster configurations
    """

    default: str = "lightgbm"
    mode: str = "single"  # "single", "auto", or "ensemble"
    training_mode: str = "individual"  # "individual" or "combined"
    per_ticker: dict[str, str] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=lambda: ["naive", "xgboost", "lightgbm"])
    selection_metric: str = "directional_accuracy"
    selection_backtest_steps: int = 30
    skip: list[str] = field(default_factory=lambda: ["nbeats", "nhits", "prophet"])
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    VALID_MODES = ("single", "auto", "ensemble")
    VALID_TRAINING_MODES = ("individual", "combined")
    VALID_SELECTION_METRICS = ("directional_accuracy", "rmse", "mae", "mape", "mase", "smape")

    def __post_init__(self):
        if self.mode not in self.VALID_MODES:
            raise ConfigValidationError(
                f"Invalid forecasters mode '{self.mode}'. Valid modes: {self.VALID_MODES}"
            )
        if self.training_mode not in self.VALID_TRAINING_MODES:
            raise ConfigValidationError(
                f"Invalid training_mode '{self.training_mode}'. "
                f"Valid modes: {self.VALID_TRAINING_MODES}"
            )
        if self.selection_metric not in self.VALID_SELECTION_METRICS:
            raise ConfigValidationError(
                f"Invalid selection_metric '{self.selection_metric}'. "
                f"Valid metrics: {self.VALID_SELECTION_METRICS}"
            )
        if self.selection_backtest_steps < 1:
            raise ConfigValidationError(
                f"selection_backtest_steps must be >= 1, got {self.selection_backtest_steps}"
            )

    def get_forecaster_for_ticker(self, ticker: str) -> str:
        """
        Get the forecaster to use for a specific ticker.

        :param ticker: Stock ticker symbol
        :return: Forecaster name (from per_ticker mapping or default)
        """
        return self.per_ticker.get(ticker, self.default)


@dataclass
class AllocatorsConfig:
    """Allocators configuration."""

    default: str = "robust_markowitz"
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class BacktestingConfig:
    """Backtesting configuration."""

    initial_train_size: int = 252
    test_size: int = 1
    step_size: int = 1
    min_train_size: int = 60
    comparison: dict[str, int] = field(default_factory=lambda: {"step_size": 20})

    def __post_init__(self):
        if self.initial_train_size < 10:
            raise ConfigValidationError(
                f"initial_train_size must be >= 10, got {self.initial_train_size}"
            )
        if self.test_size < 1:
            raise ConfigValidationError(f"test_size must be >= 1, got {self.test_size}")
        if self.step_size < 1:
            raise ConfigValidationError(f"step_size must be >= 1, got {self.step_size}")
        if self.min_train_size < 10:
            raise ConfigValidationError(f"min_train_size must be >= 10, got {self.min_train_size}")
        if self.min_train_size > self.initial_train_size:
            raise ConfigValidationError(
                f"min_train_size ({self.min_train_size}) cannot exceed "
                f"initial_train_size ({self.initial_train_size})"
            )


@dataclass
class DatabaseConfig:
    """Database configuration."""

    mode: str = "neon"
    table_name: str = "stock_optimisation_store"

    @property
    def url(self) -> str | None:
        """Get database URL from environment variable."""
        return os.getenv("DATABASE_URL")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def __post_init__(self):
        if self.level.upper() not in self.VALID_LEVELS:
            raise ConfigValidationError(
                f"Invalid logging level '{self.level}'. Valid levels: {self.VALID_LEVELS}"
            )


@dataclass
class ProphetConfig:
    """Prophet-specific configuration."""

    holiday_name_map: dict[str, str] = field(default_factory=dict)
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False


class Config:
    """
    Main configuration class.

    Loads configuration from YAML files and provides typed access to all settings.

    Profiles available:
        - default: Standard settings (config/default.yaml)
        - production: Live trading (config/production.yaml)
        - development: Dev/testing (config/development.yaml)
        - backtest: Comprehensive backtesting (config/backtest.yaml)
        - paper_trading: Simulated trading (config/paper_trading.yaml)
        - debug: Debugging issues (config/debug.yaml)
        - fast: Quick smoke tests (config/fast.yaml)
        - full_comparison: All forecasters x all allocators, combined training (config/full_comparison.yaml)

    Example:
        >>> from src.config import get_config
        >>> config = get_config()  # Uses default profile
        >>> config = get_config(profile="production")  # Uses production profile
        >>> print(config.portfolio.tickers[:5])
        ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META']
    """

    VALID_PROFILES = [
        "default",
        "production",
        "development",
        "backtest",
        "paper_trading",
        "debug",
        "fast",
        "full_comparison",
    ]

    def __init__(self, config_dir: Path | str | None = None, profile: str = "default"):
        """
        Initialize configuration.

        :param config_dir: Directory containing config files. If None, uses project's config/
        :param profile: Configuration profile to load (default, production, development, etc.)
        """
        if config_dir is None:
            # Find config directory relative to this file
            config_dir = Path(__file__).parent.parent / "config"
        else:
            config_dir = Path(config_dir)

        self._config_dir = config_dir
        self._profile = profile
        self._raw_config: dict = {}

        # Load configuration
        self._load_config()

        # Parse into typed objects
        self._parse_config()

    @property
    def profile(self) -> str:
        """Get the current configuration profile name."""
        return self._profile

    def _load_config(self) -> None:
        """
        Load configuration from YAML files.

        Loading order (later files override earlier):
        1. config/default.yaml (base defaults)
        2. config/{profile}.yaml (profile-specific settings, if profile != "default")
        3. config/local.yaml (local overrides, gitignored)
        """
        default_path = self._config_dir / "default.yaml"
        profile_path = self._config_dir / f"{self._profile}.yaml"
        local_path = self._config_dir / "local.yaml"

        # 1. Load default config (always required)
        if default_path.exists():
            with open(default_path, encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            self._raw_config = {}

        # 2. Merge profile-specific config if not default
        if self._profile != "default" and profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                profile_config = yaml.safe_load(f) or {}
                self._raw_config = _deep_merge(self._raw_config, profile_config)

        # 3. Merge local config if exists (highest priority)
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                local_config = yaml.safe_load(f) or {}
                self._raw_config = _deep_merge(self._raw_config, local_config)

    def _parse_config(self) -> None:
        """Parse raw config into typed dataclasses."""
        # Execution
        execution_data = self._raw_config.get("execution", {})
        self.execution = ExecutionConfig(
            mode=execution_data.get("mode", "single"),
            save_to_db=execution_data.get("save_to_db", True),
            horizon=execution_data.get("horizon", 21),
            verbose=execution_data.get("verbose", True),
        )

        # Portfolio
        portfolio_data = self._raw_config.get("portfolio", {})
        risk_data = portfolio_data.get("risk", {})
        self.portfolio = PortfolioConfig(
            tickers=portfolio_data.get("tickers", []),
            risk=RiskConfig(**risk_data) if risk_data else RiskConfig(),
        )

        # Dates
        dates_data = self._raw_config.get("dates", {})
        self.dates = DatesConfig(
            start_date=dates_data.get("start_date", "2024-01-01"),
            end_date=dates_data.get("end_date"),
        )

        # Forecasters
        forecasters_data = self._raw_config.get("forecasters", {})
        self.forecasters = ForecastersConfig(
            default=forecasters_data.get("default", "lightgbm"),
            mode=forecasters_data.get("mode", "single"),
            training_mode=forecasters_data.get("training_mode", "individual"),
            per_ticker=forecasters_data.get("per_ticker", {}) or {},
            candidates=forecasters_data.get("candidates", ["naive", "xgboost", "lightgbm"]),
            selection_metric=forecasters_data.get("selection_metric", "directional_accuracy"),
            selection_backtest_steps=forecasters_data.get("selection_backtest_steps", 30),
            skip=forecasters_data.get("skip", ["nbeats", "nhits", "prophet"]),
            configs=forecasters_data.get("configs", {}),
        )

        # Allocators
        allocators_data = self._raw_config.get("allocators", {})
        self.allocators = AllocatorsConfig(
            default=allocators_data.get("default", "robust_markowitz"),
            configs=allocators_data.get("configs", {}),
        )

        # Backtesting
        backtesting_data = self._raw_config.get("backtesting", {})
        self.backtesting = BacktestingConfig(
            initial_train_size=backtesting_data.get("initial_train_size", 252),
            test_size=backtesting_data.get("test_size", 1),
            step_size=backtesting_data.get("step_size", 1),
            min_train_size=backtesting_data.get("min_train_size", 60),
            comparison=backtesting_data.get("comparison", {"step_size": 20}),
        )

        # Database
        database_data = self._raw_config.get("database", {})
        self.database = DatabaseConfig(
            mode=database_data.get("mode", "neon"),
            table_name=database_data.get("table_name", "stock_optimisation_store"),
        )

        # Logging
        logging_data = self._raw_config.get("logging", {})
        self.logging = LoggingConfig(
            level=logging_data.get("level", "INFO"),
            format=logging_data.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )

        # Prophet
        prophet_data = self._raw_config.get("prophet", {})
        self.prophet = ProphetConfig(
            holiday_name_map=prophet_data.get("holiday_name_map", {}),
            yearly_seasonality=prophet_data.get("yearly_seasonality", True),
            weekly_seasonality=prophet_data.get("weekly_seasonality", True),
            daily_seasonality=prophet_data.get("daily_seasonality", False),
        )

    def get_forecaster_config(self, name: str) -> dict[str, Any]:
        """
        Get configuration for a specific forecaster.

        :param name: Forecaster name
        :return: Configuration dictionary (empty if not found)
        """
        return (self.forecasters.configs.get(name) or {}).copy()

    def get_allocator_config(self, name: str) -> dict[str, Any]:
        """
        Get configuration for a specific allocator.

        :param name: Allocator name
        :return: Configuration dictionary (empty if not found)
        """
        return (self.allocators.configs.get(name) or {}).copy()

    def should_skip_forecaster(self, name: str) -> bool:
        """
        Check if a forecaster should be skipped.

        :param name: Forecaster name
        :return: True if should be skipped
        """
        return name in self.forecasters.skip

    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()
        self._parse_config()

    @property
    def raw(self) -> dict:
        """Get raw configuration dictionary."""
        return self._raw_config.copy()


# Global config instances (lazy loading, cached by profile)
_config_cache: dict[str, Config] = {}


def get_config(profile: str = "default") -> Config:
    """
    Get a configuration instance for the specified profile.

    Profiles available:
        - default: Standard settings
        - production: Live trading settings
        - development: Development/testing settings
        - backtest: Backtesting configuration
        - paper_trading: Paper trading simulation
        - debug: Debugging configuration
        - fast: Quick smoke test configuration
        - full_comparison: All forecasters x all allocators, combined training

    :param profile: Configuration profile name
    :return: Config instance for the profile
    :raises ValueError: If profile is invalid

    Example:
        >>> config = get_config()  # Default profile
        >>> config = get_config("production")  # Production settings
        >>> config = get_config("debug")  # Debug settings
    """
    global _config_cache

    if profile not in Config.VALID_PROFILES:
        raise ValueError(f"Invalid profile '{profile}'. Valid profiles: {Config.VALID_PROFILES}")

    if profile not in _config_cache:
        _config_cache[profile] = Config(profile=profile)
    return _config_cache[profile]


def clear_config_cache() -> None:
    """Clear the configuration cache, forcing reload on next get_config() call."""
    global _config_cache
    _config_cache = {}
