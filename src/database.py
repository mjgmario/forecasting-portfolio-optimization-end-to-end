"""
Database Operations Module

Handles all database interactions for storing forecasting results,
portfolio allocations, and backtesting metrics.

Supports two database backends:
1. **SQLite** (local development): sqlite:///local.db
2. **PostgreSQL** (production/Neon): postgresql://user:password@host:port/dbname

Environment Variables:
    - DATABASE_URL: Database connection string
      - Local: sqlite:///data/forecasting.db
      - Production: postgresql://user:password@ep-example.neon.tech/neondb

Tables:
    - forecasts: Individual forecast predictions per asset
    - portfolios: Portfolio weights and metrics per date
    - backtests: Backtesting results and model comparisons
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session, scoped_session, sessionmaker

logger = logging.getLogger(__name__)


def _is_sqlite(connection_string: str) -> bool:
    """Check if connection string is for SQLite."""
    return connection_string.startswith("sqlite")


def _get_json_column():
    """Get appropriate JSON column type based on dialect."""
    # Use Text for SQLite, JSONB for PostgreSQL
    # The actual type is determined at runtime in DatabaseManager
    return Text


def _get_id_column():
    """Get appropriate ID column type based on dialect."""
    # Use String(36) for SQLite, UUID for PostgreSQL
    # The actual type is determined at runtime in DatabaseManager
    return String(36)


# Database schema definition - uses Text for JSON fields (works with both SQLite and PostgreSQL)
metadata = MetaData()

forecasts_table = Table(
    "forecasts",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("as_of_date", DateTime, nullable=False),
    Column("prediction_date", DateTime, nullable=False),
    Column("ticker", String(20), nullable=False),
    Column("forecaster", String(50), nullable=False),
    Column("predicted_price", Float, nullable=False),
    Column("predicted_return", Float, nullable=False),
    Column("actual_price", Float, nullable=True),  # Filled later when known
    Column("actual_return", Float, nullable=True),
    Column("error", Float, nullable=True),
    Column("forecaster_config", Text, nullable=True),  # JSON as Text
)

portfolios_table = Table(
    "portfolios",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("as_of_date", DateTime, nullable=False),
    Column("rebalance_date", DateTime, nullable=False),
    Column("forecaster", String(50), nullable=False),
    Column("allocator", String(50), nullable=False),
    Column("weights", Text, nullable=False),  # JSON as Text
    Column("expected_return", Float, nullable=True),
    Column("expected_volatility", Float, nullable=True),
    Column("sharpe_ratio", Float, nullable=True),
    Column("actual_return", Float, nullable=True),  # Filled later
    Column("allocator_config", Text, nullable=True),  # JSON as Text
)

backtests_table = Table(
    "backtests",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("forecaster", String(50), nullable=False),
    Column("allocator", String(50), nullable=True),
    Column("start_date", DateTime, nullable=False),
    Column("end_date", DateTime, nullable=False),
    Column("n_folds", Integer, nullable=False),
    Column("mae", Float, nullable=True),
    Column("rmse", Float, nullable=True),
    Column("mase", Float, nullable=True),
    Column("directional_accuracy", Float, nullable=True),
    Column("sharpe_ratio", Float, nullable=True),
    Column("total_return", Float, nullable=True),
    Column("max_drawdown", Float, nullable=True),
    Column("config", Text, nullable=True),  # JSON as Text
    Column("metrics", Text, nullable=True),  # JSON as Text
)


class DatabaseManager:
    """
    Manager for all database operations.

    Supports both SQLite (local development) and PostgreSQL (production/Neon).
    Handles connection pooling, session management, and CRUD operations
    for forecasts, portfolios, and backtesting results.

    :param connection_string: Database connection URL
    :type connection_string: str, optional
    :param echo: Enable SQLAlchemy query logging
    :type echo: bool

    Example:
        >>> # Local development with SQLite
        >>> db = DatabaseManager("sqlite:///data/forecasting.db")
        >>> db.create_tables()
        >>> db.save_forecast_results(results)

        >>> # Production with Neon PostgreSQL
        >>> db = DatabaseManager("postgresql://user:pass@ep-example.neon.tech/db")

    Note:
        Neon has a 0.5GB storage limit on free tier. Use cleanup_old_data()
        to remove old records and stay within limits.
    """

    def __init__(self, connection_string: str | None = None, echo: bool = False):
        """
        Initialize database manager.

        :param connection_string: Database URL (defaults to DATABASE_URL env var)
        :type connection_string: str, optional
        :param echo: Enable SQL query logging
        :type echo: bool
        :raises ValueError: If connection string not provided and DATABASE_URL not set
        """
        # Check DATABASE_MODE to determine which connection to use
        db_mode = os.getenv("DATABASE_MODE", "neon").lower()

        if connection_string:
            self.connection_string = connection_string
        elif db_mode == "local":
            # Use local SQLite
            self.connection_string = os.getenv(
                "DATABASE_URL_LOCAL", "sqlite:///data/forecasting.db"
            )
        else:
            # Use Neon PostgreSQL (default)
            self.connection_string = os.getenv("DATABASE_URL")

        if not self.connection_string:
            # Fallback to local SQLite if no DATABASE_URL
            self.connection_string = "sqlite:///data/forecasting.db"
            logger.warning("DATABASE_URL not set. Using local SQLite: data/forecasting.db")

        self.is_sqlite = _is_sqlite(self.connection_string)

        # Create engine with appropriate settings
        if self.is_sqlite:
            # SQLite: no connection pooling needed
            self.engine = create_engine(
                self.connection_string,
                echo=echo,
                connect_args={"check_same_thread": False},
            )
            # Sanitize log: only show path, not full connection string with potential creds
            db_path = self.connection_string.replace("sqlite:///", "")
            logger.info(f"Using SQLite database: {db_path}")
        else:
            # PostgreSQL: use connection pooling
            self.engine = create_engine(
                self.connection_string,
                echo=echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
            # Never log PostgreSQL connection string (contains credentials)
            logger.info("Using PostgreSQL database (connection string not logged for security)")

        self.SessionLocal = sessionmaker(bind=self.engine)
        # Thread-safe scoped session for multi-threaded usage
        self.ScopedSession = scoped_session(self.SessionLocal)

    def create_tables(self) -> None:
        """
        Create all database tables if they don't exist.

        :raises Exception: If table creation fails
        """
        try:
            metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def get_session(self, thread_safe: bool = False) -> Session:
        """
        Get a database session.

        :param thread_safe: If True, return a thread-local scoped session
        :type thread_safe: bool
        :return: SQLAlchemy session
        :rtype: Session

        Example:
            >>> with db.get_session() as session:
            ...     session.query(forecasts_table).all()

            >>> # For multi-threaded usage
            >>> session = db.get_session(thread_safe=True)
            >>> try:
            ...     # do work
            ... finally:
            ...     db.remove_scoped_session()
        """
        if thread_safe:
            return self.ScopedSession()
        return self.SessionLocal()

    def remove_scoped_session(self) -> None:
        """
        Remove the current thread's scoped session.

        Call this when done with a thread-safe session to release resources.
        Important for long-running threads or thread pools.
        """
        self.ScopedSession.remove()

    @contextmanager
    def session_scope(self, thread_safe: bool = False) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic commit/rollback.

        Provides a transactional scope around a series of operations.
        Commits on success, rolls back on exception.

        :param thread_safe: If True, use thread-local scoped session
        :type thread_safe: bool
        :yields: SQLAlchemy session
        :rtype: Generator[Session, None, None]

        Example:
            >>> with db.session_scope() as session:
            ...     session.execute(forecasts_table.insert(), rows)
            ...     # Commits automatically on success

            >>> # Multi-threaded usage
            >>> with db.session_scope(thread_safe=True) as session:
            ...     # Thread-safe session with automatic cleanup
        """
        session = self.get_session(thread_safe=thread_safe)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if thread_safe:
                self.remove_scoped_session()
            else:
                session.close()

    def save_forecast_results(
        self,
        results: dict[str, Any],
        forecaster_name: str = "prophet",
        forecaster_config: dict | None = None,
    ) -> None:
        """
        Save forecast results to database.

        :param results: Results from forecasting pipeline
        :type results: dict
        :param forecaster_name: Name of forecaster used
        :type forecaster_name: str
        :param forecaster_config: Forecaster configuration
        :type forecaster_config: dict, optional
        :raises Exception: If database insertion fails

        Example:
            >>> results = {
            ...     "date": date(2024, 1, 15),
            ...     "predictions": {"AAPL": 150.0, "MSFT": 300.0},
            ...     "predicted_returns": {"AAPL": 0.02, "MSFT": 0.015}
            ... }
            >>> db.save_forecast_results(results, "xgboost")
        """
        as_of_date = results.get("date")
        predictions = results.get("predictions", {})
        predicted_returns = results.get("predicted_returns", {})

        if not predictions:
            logger.warning("No predictions to save")
            return

        # Determine prediction date (next trading day)
        prediction_date = as_of_date  # Simplified - should use market calendar

        rows = []
        for ticker, predicted_price in predictions.items():
            row = {
                "id": str(uuid.uuid4()),  # Use string for SQLite compatibility
                "created_at": datetime.utcnow(),
                "as_of_date": as_of_date,
                "prediction_date": prediction_date,
                "ticker": ticker,
                "forecaster": forecaster_name,
                "predicted_price": float(predicted_price),
                "predicted_return": float(predicted_returns.get(ticker, 0.0)),
                "forecaster_config": json.dumps(forecaster_config) if forecaster_config else None,
            }
            rows.append(row)

        # Insert to database using session_scope for auto commit/rollback
        with self.session_scope() as session:
            try:
                session.execute(forecasts_table.insert(), rows)
                logger.info(f"Saved {len(rows)} forecast records to database")
            except Exception as e:
                logger.error(f"Failed to save forecasts: {e}")
                raise

    def save_portfolio_results(
        self,
        results: dict[str, Any],
        forecaster_name: str,
        allocator_name: str,
        allocator_config: dict | None = None,
    ) -> None:
        """
        Save portfolio allocation results to database.

        :param results: Results from portfolio optimization
        :type results: dict
        :param forecaster_name: Name of forecaster used
        :type forecaster_name: str
        :param allocator_name: Name of allocator used
        :type allocator_name: str
        :param allocator_config: Allocator configuration
        :type allocator_config: dict, optional
        :raises Exception: If database insertion fails

        Example:
            >>> results = {
            ...     "date": date(2024, 1, 15),
            ...     "weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
            ... }
            >>> db.save_portfolio_results(results, "xgboost", "robust_markowitz")
        """
        as_of_date = results.get("date")
        weights = results.get("weights", {})

        # Handle empty weights (works for dict, Series, or DataFrame)
        if weights is None or (hasattr(weights, "__len__") and len(weights) == 0):
            logger.warning("No portfolio weights to save")
            return

        # Convert weights to JSON-serializable dict (handle pandas Series/numpy types)
        weights_serializable = {}
        # If it's a pandas Series, iterate properly
        if hasattr(weights, "items"):
            for k, v in weights.items():
                if hasattr(v, "item"):  # numpy type
                    weights_serializable[str(k)] = v.item()
                else:
                    weights_serializable[str(k)] = float(v)
        else:
            weights_serializable = {str(k): float(v) for k, v in weights.items()}

        row = {
            "id": str(uuid.uuid4()),  # Use string for SQLite compatibility
            "created_at": datetime.utcnow(),
            "as_of_date": as_of_date,
            "rebalance_date": as_of_date,  # Simplified
            "forecaster": forecaster_name,
            "allocator": allocator_name,
            "weights": json.dumps(weights_serializable),  # Serialize to JSON string
            "allocator_config": json.dumps(allocator_config) if allocator_config else None,
        }

        # Use session_scope for auto commit/rollback
        with self.session_scope() as session:
            try:
                session.execute(portfolios_table.insert(), [row])
                logger.info("Saved portfolio allocation to database")
            except Exception as e:
                logger.error(f"Failed to save portfolio: {e}")
                raise

    def save_backtest_results(
        self,
        forecaster: str,
        metrics: dict[str, float],
        start_date: datetime,
        end_date: datetime,
        n_folds: int,
        config: dict | None = None,
        allocator: str | None = None,
    ) -> None:
        """
        Save backtesting results to database.

        :param forecaster: Name of forecaster
        :type forecaster: str
        :param metrics: Dictionary of evaluation metrics
        :type metrics: dict
        :param start_date: Backtest start date
        :type start_date: datetime
        :param end_date: Backtest end date
        :type end_date: datetime
        :param n_folds: Number of backtest folds
        :type n_folds: int
        :param config: Configuration used for backtest
        :type config: dict, optional
        :param allocator: Name of allocator (if portfolio backtest)
        :type allocator: str, optional
        :raises Exception: If database insertion fails
        """
        # Serialize metrics dict (convert numpy types to native Python)
        metrics_serializable = {
            k: float(v) if v is not None else None
            for k, v in metrics.items()
            if isinstance(v, (int, float, type(None)))
        }

        row = {
            "id": str(uuid.uuid4()),  # Use string for SQLite compatibility
            "created_at": datetime.utcnow(),
            "forecaster": forecaster,
            "allocator": allocator,
            "start_date": start_date,
            "end_date": end_date,
            "n_folds": n_folds,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mase": metrics.get("mase"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "config": json.dumps(config) if config else None,
            "metrics": json.dumps(metrics_serializable),  # Serialize to JSON string
        }

        # Use session_scope for auto commit/rollback
        with self.session_scope() as session:
            try:
                session.execute(backtests_table.insert(), [row])
                logger.info(f"Saved backtest results for {forecaster}")
            except Exception as e:
                logger.error(f"Failed to save backtest results: {e}")
                raise

    def get_latest_forecasts(self, n: int = 10) -> pd.DataFrame:
        """
        Retrieve latest forecast records.

        :param n: Number of records to retrieve
        :type n: int
        :return: DataFrame with forecast records
        :rtype: pd.DataFrame
        """
        query = """
            SELECT * FROM forecasts
            ORDER BY created_at DESC
            LIMIT :n
        """
        # Use engine.connect() for proper connection management with pandas
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"n": n})

    def get_latest_portfolios(self, n: int = 10) -> pd.DataFrame:
        """
        Retrieve latest portfolio allocations.

        :param n: Number of records to retrieve
        :type n: int
        :return: DataFrame with portfolio records
        :rtype: pd.DataFrame
        """
        query = """
            SELECT * FROM portfolios
            ORDER BY created_at DESC
            LIMIT :n
        """
        # Use engine.connect() for proper connection management with pandas
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"n": n})

    def get_backtest_results(self, forecaster: str | None = None) -> pd.DataFrame:
        """
        Retrieve backtest results, optionally filtered by forecaster.

        :param forecaster: Filter by forecaster name
        :type forecaster: str, optional
        :return: DataFrame with backtest results
        :rtype: pd.DataFrame
        """
        params: dict[str, str] = {}
        if forecaster:
            query = (
                "SELECT * FROM backtests WHERE forecaster = :forecaster ORDER BY created_at DESC"
            )
            params["forecaster"] = forecaster
        else:
            query = "SELECT * FROM backtests ORDER BY created_at DESC"

        # Use engine.connect() for proper connection management with pandas
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params=params if params else None)

    def cleanup_old_data(self, days_to_keep: int = 30) -> dict[str, int]:
        """
        Remove old records to stay within Neon's 0.5GB storage limit.

        :param days_to_keep: Number of days of data to retain
        :type days_to_keep: int
        :return: Dictionary with count of deleted records per table
        :rtype: dict

        Example:
            >>> db = get_db()
            >>> deleted = db.cleanup_old_data(days_to_keep=30)
            >>> print(f"Deleted {deleted['forecasts']} forecast records")
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted_counts = {"forecasts": 0, "portfolios": 0, "backtests": 0}

        # Use session_scope for auto commit/rollback
        with self.session_scope() as session:
            try:
                # Delete old forecasts
                result = session.execute(
                    forecasts_table.delete().where(forecasts_table.c.created_at < cutoff_date)
                )
                deleted_counts["forecasts"] = result.rowcount

                # Delete old portfolios
                result = session.execute(
                    portfolios_table.delete().where(portfolios_table.c.created_at < cutoff_date)
                )
                deleted_counts["portfolios"] = result.rowcount

                # Delete old backtests
                result = session.execute(
                    backtests_table.delete().where(backtests_table.c.created_at < cutoff_date)
                )
                deleted_counts["backtests"] = result.rowcount

                logger.info(
                    f"Cleaned up old data (>{days_to_keep} days): "
                    f"forecasts={deleted_counts['forecasts']}, "
                    f"portfolios={deleted_counts['portfolios']}, "
                    f"backtests={deleted_counts['backtests']}"
                )

            except Exception as e:
                logger.error(f"Failed to cleanup old data: {e}")
                raise

        return deleted_counts

    def get_database_size(self) -> dict[str, Any]:
        """
        Get current database size information (PostgreSQL only).

        :return: Dictionary with size information
        :rtype: dict

        Example:
            >>> db = get_db()
            >>> size_info = db.get_database_size()
            >>> print(f"Database size: {size_info['total_size_mb']} MB")
        """
        if self.is_sqlite:
            # For SQLite, get file size
            import os as os_module

            db_path = self.connection_string.replace("sqlite:///", "")
            if os_module.path.exists(db_path):
                size_bytes = os_module.path.getsize(db_path)
                return {
                    "total_size_mb": size_bytes / (1024 * 1024),
                    "total_size_bytes": size_bytes,
                    "type": "sqlite",
                }
            return {"total_size_mb": 0, "type": "sqlite", "error": "File not found"}

        # PostgreSQL size query - use engine.connect() for proper connection handling
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text("SELECT pg_database_size(current_database()) as size"))
                row = result.fetchone()
                size_bytes = row[0] if row else 0
                return {
                    "total_size_mb": size_bytes / (1024 * 1024),
                    "total_size_bytes": size_bytes,
                    "limit_mb": 512,  # Neon free tier limit
                    "usage_percent": (size_bytes / (512 * 1024 * 1024)) * 100,
                    "type": "postgresql",
                }
            except Exception as e:
                logger.warning(f"Could not get database size: {e}")
                return {"error": str(e), "type": "postgresql"}


# Singleton instance for easy access
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """
    Get or create DatabaseManager singleton.

    :return: DatabaseManager instance
    :rtype: DatabaseManager
    :raises ValueError: If DATABASE_URL not set

    Example:
        >>> db = get_db()
        >>> db.save_forecast_results(results)
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


__all__ = [
    "DatabaseManager",
    "get_db",
    "forecasts_table",
    "portfolios_table",
    "backtests_table",
]
