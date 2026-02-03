"""
Constants Module

Centralized constants used throughout the forecasting system.
This eliminates magic numbers and provides a single source of truth.
"""

# =============================================================================
# TRADING CALENDAR
# =============================================================================

# Standard trading days per year (US markets)
TRADING_DAYS_PER_YEAR = 252

# Standard trading days per month (approximate)
TRADING_DAYS_PER_MONTH = 21

# Standard trading days per week
TRADING_DAYS_PER_WEEK = 5


# =============================================================================
# DATE/TIME FREQUENCIES (pandas freq strings)
# =============================================================================

# Business days (Mon-Fri, excluding market holidays)
FREQ_BUSINESS_DAY = "B"

# Calendar days (all days)
FREQ_CALENDAR_DAY = "D"

# Weekly frequency
FREQ_WEEKLY = "W"

# Monthly frequency (end of month)
FREQ_MONTHLY = "ME"

# Default frequency for financial time series
FREQ_DEFAULT = FREQ_BUSINESS_DAY


# =============================================================================
# FORECASTING DEFAULTS
# =============================================================================

# Default forecast horizon (days)
DEFAULT_HORIZON = 21

# Maximum allowed forecast horizon
MAX_HORIZON = 365

# Default number of lags for ML models
DEFAULT_LAGS = 5

# Default rolling window sizes for feature engineering
DEFAULT_WINDOWS = [5, 20]

# Minimum data points required for training
MIN_TRAINING_SAMPLES = 50


# =============================================================================
# ML MODEL DEFAULTS
# =============================================================================

# Train/validation split ratio for early stopping
EARLY_STOPPING_SPLIT = 0.8

# Minimum samples required for early stopping validation
MIN_EARLY_STOPPING_SAMPLES = 50

# Default number of estimators for ensemble models
DEFAULT_N_ESTIMATORS = 100

# Default learning rate for gradient boosting models
DEFAULT_LEARNING_RATE = 0.1

# Default max depth for tree-based models
DEFAULT_MAX_DEPTH = 6

# Default early stopping rounds
DEFAULT_EARLY_STOPPING_ROUNDS = 10


# =============================================================================
# BACKTESTING DEFAULTS
# =============================================================================

# Default initial training size (1 year of trading data)
DEFAULT_INITIAL_TRAIN_SIZE = TRADING_DAYS_PER_YEAR

# Default test size for walk-forward validation
DEFAULT_TEST_SIZE = 1

# Default step size for walk-forward validation
DEFAULT_STEP_SIZE = 1

# Default gap size (embargo period) between train and test
DEFAULT_GAP_SIZE = 0

# Default refit frequency (refit model every N steps)
DEFAULT_REFIT_EVERY = 1


# =============================================================================
# PORTFOLIO DEFAULTS
# =============================================================================

# Default minimum allocation per asset
DEFAULT_MIN_ALLOCATION = 0.0

# Default maximum allocation per asset
DEFAULT_MAX_ALLOCATION = 1.0

# Default risk aversion parameter
DEFAULT_RISK_AVERSION = 1.0

# Default risk-free rate (annualized)
DEFAULT_RISK_FREE_RATE = 0.0

# Annualization factor for Sharpe ratio (daily to annual)
ANNUALIZATION_FACTOR = TRADING_DAYS_PER_YEAR**0.5


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Small epsilon for numerical stability
EPSILON = 1e-10

# Default covariance shrinkage intensity
DEFAULT_COV_SHRINKAGE = 0.1

# Regularization factor for matrix inversion
DEFAULT_REGULARIZATION = 1e-6


# =============================================================================
# DATA EXTRACTION
# =============================================================================

# Default years of historical data to fetch
DEFAULT_HISTORY_YEARS = 5

# Buffer days for data extraction
DATA_EXTRACTION_BUFFER_DAYS = 10


__all__ = [
    # Trading calendar
    "TRADING_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_WEEK",
    # Frequencies
    "FREQ_BUSINESS_DAY",
    "FREQ_CALENDAR_DAY",
    "FREQ_WEEKLY",
    "FREQ_MONTHLY",
    "FREQ_DEFAULT",
    # Forecasting
    "DEFAULT_HORIZON",
    "MAX_HORIZON",
    "DEFAULT_LAGS",
    "DEFAULT_WINDOWS",
    "MIN_TRAINING_SAMPLES",
    # ML models
    "EARLY_STOPPING_SPLIT",
    "MIN_EARLY_STOPPING_SAMPLES",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_EARLY_STOPPING_ROUNDS",
    # Backtesting
    "DEFAULT_INITIAL_TRAIN_SIZE",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_STEP_SIZE",
    "DEFAULT_GAP_SIZE",
    "DEFAULT_REFIT_EVERY",
    # Portfolio
    "DEFAULT_MIN_ALLOCATION",
    "DEFAULT_MAX_ALLOCATION",
    "DEFAULT_RISK_AVERSION",
    "DEFAULT_RISK_FREE_RATE",
    "ANNUALIZATION_FACTOR",
    # Numerical
    "EPSILON",
    "DEFAULT_COV_SHRINKAGE",
    "DEFAULT_REGULARIZATION",
    # Data extraction
    "DEFAULT_HISTORY_YEARS",
    "DATA_EXTRACTION_BUFFER_DAYS",
]
