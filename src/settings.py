"""
Settings and constants for portfolio optimisation.

This module provides backward-compatible access to configuration.
All settings are now loaded from config/default.yaml (or config/local.yaml).

For new code, prefer using:
    >>> from src.config import get_config
    >>> config = get_config()
    >>> tickers = config.portfolio.tickers

Legacy usage (still supported):
    >>> from src.settings import PORTFOLIO_TICKERS, START_DATE
"""

from typing import Any

from .config import get_config


class _Settings:
    """
    Settings container that wraps config for backward compatibility.

    This class provides attribute-based access to configuration values
    and supports reloading without needing global declarations.
    """

    def __init__(self):
        self._config = get_config()
        self._update_from_config()

    def _update_from_config(self) -> None:
        """Update all settings from the config object."""
        cfg = self._config

        # Risk parameters
        self.MINIMUM_ALLOCATION = cfg.portfolio.risk.minimum_allocation
        self.MAXIMUM_ALLOCATION = cfg.portfolio.risk.maximum_allocation
        self.RISK_AVERSION = cfg.portfolio.risk.risk_aversion

        # Date defaults
        self.START_DATE = cfg.dates.start_date
        self.END_DATE = cfg.dates.effective_end_date

        # Stock Allocation
        self.PORTFOLIO_TICKERS = cfg.portfolio.tickers

        # Database
        self.SUPABASE_TABLE_NAME = cfg.database.table_name

        # Prophet model configuration
        self.HOLIDAY_NAME_MAP = cfg.prophet.holiday_name_map
        self.PROPHET_PARAMS = {
            "yearly_seasonality": cfg.prophet.yearly_seasonality,
            "weekly_seasonality": cfg.prophet.weekly_seasonality,
            "daily_seasonality": cfg.prophet.daily_seasonality,
        }

        # Backtesting configuration
        self.BACKTEST_INITIAL_TRAIN_SIZE = cfg.backtesting.initial_train_size
        self.BACKTEST_TEST_SIZE = cfg.backtesting.test_size
        self.BACKTEST_STEP_SIZE = cfg.backtesting.step_size
        self.BACKTEST_MIN_TRAIN_SIZE = cfg.backtesting.min_train_size

        # Forecaster configuration
        self.SKIP_FORECASTERS = cfg.forecasters.skip
        self.DEFAULT_FORECASTER = cfg.forecasters.default
        self.DEFAULT_ALLOCATOR = cfg.allocators.default

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config.reload()
        self._update_from_config()

    def get_forecaster_config(self, name: str) -> dict[str, Any]:
        """Get configuration for a specific forecaster."""
        return self._config.get_forecaster_config(name)

    def get_allocator_config(self, name: str) -> dict[str, Any]:
        """Get configuration for a specific allocator."""
        return self._config.get_allocator_config(name)


# Singleton instance
_settings = _Settings()

# =============================================================================
# Backward-compatible exports (module-level access)
# =============================================================================

# Risk parameters
MINIMUM_ALLOCATION = _settings.MINIMUM_ALLOCATION
MAXIMUM_ALLOCATION = _settings.MAXIMUM_ALLOCATION
RISK_AVERSION = _settings.RISK_AVERSION

# Date defaults
START_DATE = _settings.START_DATE
END_DATE = _settings.END_DATE

# Stock Allocation
PORTFOLIO_TICKERS = _settings.PORTFOLIO_TICKERS

# Database
SUPABASE_TABLE_NAME = _settings.SUPABASE_TABLE_NAME

# Prophet model configuration
HOLIDAY_NAME_MAP = _settings.HOLIDAY_NAME_MAP
PROPHET_PARAMS = _settings.PROPHET_PARAMS

# Backtesting configuration
BACKTEST_INITIAL_TRAIN_SIZE = _settings.BACKTEST_INITIAL_TRAIN_SIZE
BACKTEST_TEST_SIZE = _settings.BACKTEST_TEST_SIZE
BACKTEST_STEP_SIZE = _settings.BACKTEST_STEP_SIZE
BACKTEST_MIN_TRAIN_SIZE = _settings.BACKTEST_MIN_TRAIN_SIZE

# Forecaster configuration
SKIP_FORECASTERS = _settings.SKIP_FORECASTERS
DEFAULT_FORECASTER = _settings.DEFAULT_FORECASTER
DEFAULT_ALLOCATOR = _settings.DEFAULT_ALLOCATOR


def get_forecaster_config(name: str) -> dict[str, Any]:
    """
    Get configuration for a specific forecaster.

    :param name: Forecaster name (e.g., 'xgboost', 'arima')
    :return: Configuration dictionary
    """
    return _settings.get_forecaster_config(name)


def get_allocator_config(name: str) -> dict[str, Any]:
    """
    Get configuration for a specific allocator.

    :param name: Allocator name (e.g., 'robust_markowitz')
    :return: Configuration dictionary
    """
    return _settings.get_allocator_config(name)


def reload_config() -> None:
    """
    Reload configuration from files.

    Useful when config files have been modified during runtime.

    Note: Module-level constants (PORTFOLIO_TICKERS, etc.) are not updated
    when calling this function. For dynamic config access, use:
        from src.config import get_config
        config = get_config()
        config.reload()
    """
    _settings.reload()
