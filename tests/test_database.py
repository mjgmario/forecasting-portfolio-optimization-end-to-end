"""
Tests for Database Module

Tests database connection, table creation, and CRUD operations
using a temporary SQLite database.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.database import DatabaseManager


@pytest.fixture
def temp_db_path(request):
    """Create a temporary database file path unique per test."""
    import uuid

    # Use a unique file per test to avoid cross-test contamination
    test_id = str(uuid.uuid4())[:8]
    db_path = Path(tempfile.gettempdir()) / f"test_forecasting_{test_id}.db"
    connection_string = f"sqlite:///{db_path}"
    yield connection_string
    # Cleanup - remove the file after the test
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            pass  # Ignore on Windows if still locked


@pytest.fixture
def db_manager(temp_db_path):
    """Create a DatabaseManager with temporary SQLite database."""
    manager = DatabaseManager(connection_string=temp_db_path, echo=False)
    manager.create_tables()
    yield manager
    # Dispose engine to release file handles
    manager.engine.dispose()


class TestDatabaseConnection:
    """Test database connection and initialization."""

    def test_sqlite_connection(self, temp_db_path):
        """Test connecting to SQLite database."""
        manager = DatabaseManager(connection_string=temp_db_path)

        assert manager.is_sqlite is True
        assert manager.connection_string == temp_db_path
        assert manager.engine is not None

    def test_create_tables(self, temp_db_path):
        """Test creating database tables."""
        manager = DatabaseManager(connection_string=temp_db_path)
        manager.create_tables()

        # Verify tables exist by inspecting metadata
        with manager.engine.connect():
            # Check that we can query the tables
            from sqlalchemy import inspect

            inspector = inspect(manager.engine)
            tables = inspector.get_table_names()

            assert "forecasts" in tables
            assert "portfolios" in tables
            assert "backtests" in tables

    def test_get_session(self, db_manager):
        """Test getting a database session."""
        session = db_manager.get_session()

        assert session is not None
        session.close()

    def test_fallback_to_local_sqlite(self, monkeypatch):
        """Test fallback to local SQLite when DATABASE_URL not set."""
        # Remove DATABASE_URL if set
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DATABASE_MODE", "local")

        manager = DatabaseManager()

        assert manager.is_sqlite is True
        assert "sqlite" in manager.connection_string


class TestForecastOperations:
    """Test forecast-related database operations."""

    def test_save_forecast_results(self, db_manager):
        """Test saving forecast results."""
        results = {
            "date": datetime(2024, 1, 15),
            "predictions": {"AAPL": 150.0, "MSFT": 300.0},
            "predicted_returns": {"AAPL": 0.02, "MSFT": 0.015},
        }

        db_manager.save_forecast_results(
            results=results,
            forecaster_name="ridge",
            forecaster_config={"lags": 10},
        )

        # Verify data was saved
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("SELECT COUNT(*) FROM forecasts"))
            count = result.scalar()
            assert count == 2  # Two tickers

    def test_save_empty_results(self, db_manager):
        """Test that empty results don't cause errors."""
        results = {
            "date": datetime(2024, 1, 15),
            "predictions": {},
            "predicted_returns": {},
        }

        # Should not raise
        db_manager.save_forecast_results(results, "naive")

        # Verify nothing was saved
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("SELECT COUNT(*) FROM forecasts"))
            count = result.scalar()
            assert count == 0


class TestPortfolioOperations:
    """Test portfolio-related database operations."""

    def test_save_portfolio_results(self, db_manager):
        """Test saving portfolio allocation results."""
        portfolio_results = {
            "date": datetime(2024, 1, 15),
            "weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
            "expected_return": 0.15,
            "expected_volatility": 0.20,
            "sharpe_ratio": 0.75,
        }

        db_manager.save_portfolio_results(
            results=portfolio_results,
            forecaster_name="lightgbm",
            allocator_name="robust_markowitz",
        )

        # Verify data was saved
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("SELECT COUNT(*) FROM portfolios"))
            count = result.scalar()
            assert count == 1


class TestBacktestOperations:
    """Test backtesting-related database operations."""

    def test_save_backtest_results(self, db_manager):
        """Test saving backtesting results."""
        db_manager.save_backtest_results(
            forecaster="xgboost",
            metrics={
                "mae": 0.015,
                "rmse": 0.022,
                "directional_accuracy": 0.55,
            },
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            n_folds=10,
            config={"lags": 10},
            allocator="hrp",
        )

        # Verify data was saved
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("SELECT COUNT(*) FROM backtests"))
            count = result.scalar()
            assert count == 1


class TestDatabaseHealth:
    """Test database health and maintenance operations."""

    def test_database_is_healthy(self, db_manager):
        """Test basic database health check."""
        # Simple health check - can we execute a query?
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_multiple_sessions(self, db_manager):
        """Test that multiple sessions can be created and used."""
        sessions = [db_manager.get_session() for _ in range(3)]

        for session in sessions:
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            session.close()


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_full_pipeline_persistence(self, db_manager):
        """Test saving and retrieving data from full pipeline."""
        # 1. Save forecast
        forecast_results = {
            "date": datetime(2024, 1, 15),
            "predictions": {"AAPL": 150.0},
            "predicted_returns": {"AAPL": 0.02},
        }
        db_manager.save_forecast_results(forecast_results, "ridge")

        # 2. Save portfolio
        portfolio_results = {
            "date": datetime(2024, 1, 15),
            "weights": {"AAPL": 1.0},
            "expected_return": 0.15,
            "expected_volatility": 0.20,
            "sharpe_ratio": 0.75,
        }
        db_manager.save_portfolio_results(portfolio_results, "ridge", "equal_weight")

        # 3. Save backtest
        db_manager.save_backtest_results(
            forecaster="ridge",
            metrics={"mae": 0.01, "directional_accuracy": 0.55},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            n_folds=5,
            allocator="equal_weight",
        )

        # Verify all data
        with db_manager.get_session() as session:
            from sqlalchemy import text

            forecasts = session.execute(text("SELECT COUNT(*) FROM forecasts")).scalar()
            portfolios = session.execute(text("SELECT COUNT(*) FROM portfolios")).scalar()
            backtests = session.execute(text("SELECT COUNT(*) FROM backtests")).scalar()

            assert forecasts == 1
            assert portfolios == 1
            assert backtests == 1
