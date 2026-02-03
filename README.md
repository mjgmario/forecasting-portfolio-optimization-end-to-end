# Portfolio Forecasting & Optimization System

> **Work in Progress**: This project is under active development. Features and APIs may change.

This system pulls historical stock data, generates return forecasts using a variety of statistical, machine learning, and deep learning models, and then feeds those predictions into portfolio allocation strategies to construct optimized portfolios. The full pipeline—from data extraction through backtesting and evaluation—runs end-to-end, allowing systematic comparison of different forecasting and allocation approaches.

A comprehensive end-to-end machine learning system for stock forecasting and portfolio optimization with **19 forecasting models** and **14 allocation strategies**.

> **Disclaimer**: This project is for educational and illustrative purposes only. Not financial advice.

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run with default configuration
uv run python -m src.main

# Run with specific forecaster/allocator
uv run python -m src.main --forecaster xgboost --allocator robust_markowitz

# List available models
uv run python -m src.main --list-forecasters
uv run python -m src.main --list-allocators

# Run Streamlit dashboard
uv run streamlit run src/streamlit_dashboard.py
```

## Configuration

All settings are centralized in `config/default.yaml`. Create `config/local.yaml` for personal overrides (gitignored).

### Key Configuration Sections

```yaml
# config/default.yaml

# Portfolio tickers
portfolio:
  tickers:
    - AAPL
    - MSFT
    - NVDA
    # ... more tickers
  risk:
    minimum_allocation: 0.05
    maximum_allocation: 1.0
    risk_aversion: 5

# Forecaster settings
forecasters:
  default: xgboost
  skip:
    - nbeats   # Skip expensive deep learning models
    - nhits
  configs:
    xgboost:
      n_estimators: 200
      learning_rate: 0.05
      max_depth: 6

# Allocator settings
allocators:
  default: robust_markowitz
  configs:
    robust_markowitz:
      risk_aversion: 1.0
      covariance_shrinkage: 0.1

# Backtesting settings
backtesting:
  initial_train_size: 252  # ~1 year
  test_size: 1
  step_size: 1
```

## Available Models

### Forecasters (19 models)

| Family | Models | Description |
|--------|--------|-------------|
| **Baseline** | `naive`, `seasonal_naive`, `drift` | Simple benchmarks |
| **Statistical** | `ets`, `theta`, `arima`, `prophet` | Classical time series |
| **Machine Learning** | `ridge`, `lasso`, `elasticnet`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost` | Supervised learning |
| **Volatility** | `garch`, `egarch`, `gjr_garch` | Volatility forecasting |
| **Deep Learning** | `nbeats`, `nhits` | Neural networks (computationally expensive) |

### Allocators (14 strategies)

| Family | Strategies | Description |
|--------|------------|-------------|
| **Simple** | `equal_weight`, `inverse_volatility` | Basic allocation |
| **Risk-Based** | `risk_parity`, `minimum_variance`, `maximum_diversification` | Risk-focused |
| **Mean-Variance** | `markowitz`, `robust_markowitz`, `max_sharpe` | Classic optimization |
| **Advanced** | `black_litterman`, `hrp`, `cvar`, `kelly_criterion`, `mean_cvar`, `omega_ratio` | Sophisticated methods |

## CLI Commands

```bash
# Single forecast run (uses config defaults)
uv run python -m src.main

# Specify forecaster and allocator
uv run python -m src.main --forecaster xgboost --allocator risk_parity

# Custom tickers
uv run python -m src.main --tickers AAPL MSFT NVDA

# Compare all forecasters
uv run python -m src.main --compare-forecasters

# Comprehensive analysis (all models)
uv run python -m src.main --comprehensive --horizon 21

# Use custom config
uv run python -m src.main --config config/my_config

# Skip database save
uv run python -m src.main --no-save
```

## Architecture

```
Data Pipeline:
  Yahoo Finance -> Extractor -> Processor -> Forecaster -> Allocator -> Database
                                                ↓
                                         Streamlit Dashboard
```

### Project Structure

```
forecasting_end_to_end/
├── config/
│   └── default.yaml        # Main configuration file
├── src/
│   ├── config.py           # Configuration loader
│   ├── main.py             # CLI entry point
│   ├── forecasters/        # All forecasting models
│   │   ├── base.py         # BaseForecaster abstract class
│   │   ├── registry.py     # ForecasterRegistry
│   │   ├── baselines.py    # Naive, Drift, etc.
│   │   ├── statistical.py  # ETS, Theta, ARIMA
│   │   ├── prophet.py      # Prophet forecaster
│   │   ├── ml_regressors.py # ML models
│   │   ├── volatility.py   # GARCH models
│   │   └── deep_learning.py # NBEATS, NHITS
│   ├── allocators/         # Portfolio allocators
│   │   ├── base.py         # BaseAllocator
│   │   ├── registry.py     # AllocatorRegistry
│   │   └── models.py       # All allocator implementations
│   ├── evaluation/         # Backtesting and metrics
│   │   ├── backtesting.py  # Walk-forward validation
│   │   └── metrics.py      # MAE, RMSE, MASE, etc.
│   ├── extractor.py        # Data extraction (yfinance)
│   ├── processor.py        # Data preprocessing
│   ├── database.py         # Neon PostgreSQL
│   └── streamlit_dashboard.py
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

### Design Patterns

- **Registry Pattern**: Plugin-based model registration for forecasters and allocators
- **Strategy Pattern**: Interchangeable forecasting and allocation strategies
- **Factory Pattern**: `ForecasterRegistry.create("xgboost")`

## Backtesting

The system uses **walk-forward backtesting** (rolling window) which:
1. Avoids look-ahead bias
2. Uses only past data for training
3. Tests on future data
4. Rolls forward and repeats

```python
# Backtest configuration in config/default.yaml
backtesting:
  initial_train_size: 252   # Training window (~1 year)
  test_size: 1              # Predict 1 day ahead
  step_size: 1              # Roll forward 1 day
```

## Database Setup (Neon PostgreSQL)

1. Create a Neon account at https://neon.tech
2. Copy connection string to `.env`:

```env
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require
```

3. Initialize schema:
```bash
psql $DATABASE_URL -f scripts/setup_neon.sql
```

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Lower is better |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| **MASE** | MAE / MAE_naive | < 1 beats naive |
| **Directional Accuracy** | % correct direction | > 55% is good |

## Example Usage

```python
from src.forecasters import ForecasterRegistry
from src.allocators import AllocatorRegistry
from src.extractor import extract_data
from src.processor import preprocess_data

# 1. Extract data
data = extract_data(["AAPL", "MSFT", "NVDA"], start_date="2024-01-01")
portfolio_data = preprocess_data(data)

# 2. Forecast
forecaster = ForecasterRegistry.create("xgboost", {"n_estimators": 200})
forecaster.fit(portfolio_data["AAPL"]["Price"])
prediction = forecaster.predict(horizon=21)

# 3. Allocate
allocator = AllocatorRegistry.create("robust_markowitz")
weights = allocator.allocate(predicted_returns, historical_returns)
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format src/ tests/

# Validate system
uv run python scripts/validate_system.py
```

## GitHub Actions

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `ci.yml` | On push/PR | Tests and linting |
| `daily_forecast.yml` | 9 AM UTC (weekdays) | Daily forecasting |
| `deploy_streamlit.yml` | Manual | Dashboard deployment |

## License

MIT License - See LICENSE file for details.
