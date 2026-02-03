"""
Data Extraction Module
======================

Fetches historical stock data from Yahoo Finance using yfinance.

This module provides the data extraction layer of the pipeline:

    Yahoo Finance --> extractor.py --> processor.py --> forecasters

Usage:
    >>> from src.extractor import extract_data
    >>> data = extract_data(["AAPL", "MSFT"], start_date="2024-01-01")
    >>> print(data["AAPL"].head())
"""

import logging
import re
import time

import pandas as pd
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)

# Retry configuration for network requests
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
RETRY_BACKOFF_MULTIPLIER = 2.0

# Ticker validation pattern (1-5 uppercase letters, optionally with . or -)
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


def _validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format.

    :param ticker: Ticker symbol to validate
    :type ticker: str
    :return: True if valid, False otherwise
    :rtype: bool
    """
    if not ticker or not isinstance(ticker, str):
        return False
    return bool(TICKER_PATTERN.match(ticker.upper()))


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw ticker DataFrame from yfinance.

    Extracts close price, calculates daily returns, and normalizes dates.

    :param df: Raw DataFrame from yfinance with DatetimeIndex and 'Close' column
    :type df: pd.DataFrame
    :return: Processed DataFrame with 'Price' and 'Returns' columns
    :rtype: pd.DataFrame

    Example:
        >>> raw_df = yf.Ticker("AAPL").history(period="1mo")
        >>> processed = _process_ticker_dataframe(raw_df)
        >>> processed.columns.tolist()
        ['Price', 'Returns']
    """
    # Extract close price and rename
    df = df[["Close"]].rename(columns={"Close": "Price"})

    # Calculate daily returns (percentage change)
    df["Returns"] = df["Price"].pct_change()

    # Drop first row (NaN return)
    df = df.dropna()

    # Convert DatetimeIndex to date for consistency
    df.index = df.index.date
    df.index.name = "Date"

    return df


def _extract_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Extract and process data for a single ticker with retry logic.

    Implements exponential backoff for network failures.

    :param ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
    :type ticker: str
    :param start_date: Start date in YYYY-MM-DD format
    :type start_date: str
    :param end_date: End date in YYYY-MM-DD format
    :type end_date: str
    :return: Processed DataFrame or None if extraction fails
    :rtype: pd.DataFrame or None

    Example:
        >>> df = _extract_single_ticker("AAPL", "2024-01-01", "2024-12-31")
        >>> df is not None
        True
    """
    # Validate ticker format before making network request
    if not _validate_ticker(ticker):
        logger.warning(f"Invalid ticker format: {ticker}")
        return None

    last_exception: Exception | None = None
    delay = RETRY_DELAY_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data available for ticker: {ticker}")
                return None

            return _process_ticker_dataframe(df)

        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {ticker}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= RETRY_BACKOFF_MULTIPLIER
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {ticker}: {last_exception}")

    return None


def extract_data(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Extract historical stock data for multiple tickers.

    This is the main entry point for data extraction. It fetches data from
    Yahoo Finance for each ticker and returns processed DataFrames with
    price and returns.

    :param tickers: List of stock ticker symbols
    :type tickers: list[str]
    :param start_date: Start date in YYYY-MM-DD format (default: from config)
    :type start_date: str, optional
    :param end_date: End date in YYYY-MM-DD format (default: from config)
    :type end_date: str, optional
    :return: Dictionary mapping ticker symbol to processed DataFrame
    :rtype: dict[str, pd.DataFrame]

    Each DataFrame contains:
        - **Price**: Closing price
        - **Returns**: Daily percentage returns
        - **Index**: Date (datetime.date objects)

    Example:
        >>> from src.extractor import extract_data
        >>> data = extract_data(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31"
        ... )
        >>> "AAPL" in data
        True
        >>> data["AAPL"].columns.tolist()
        ['Price', 'Returns']

    Pipeline Integration:
        This function is typically called by main.py:

        1. **extract_data()** - Fetch raw data from Yahoo Finance
        2. **preprocess_data()** - Clean and prepare for forecasting
        3. **ForecasterRegistry.create()** - Generate predictions
        4. **AllocatorRegistry.create()** - Optimize portfolio
    """
    # Use defaults from config if not provided
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE

    logger.info(f"Extracting data for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_stock_data: dict[str, pd.DataFrame] = {}
    failed_tickers: list[str] = []

    for ticker in tickers:
        df = _extract_single_ticker(ticker, start_date, end_date)
        if df is not None:
            all_stock_data[ticker] = df
            logger.debug(f"  {ticker}: {len(df)} observations")
        else:
            failed_tickers.append(ticker)

    # Summary logging
    success_count = len(all_stock_data)
    fail_count = len(failed_tickers)

    if fail_count > 0:
        logger.warning(f"Failed to extract {fail_count} tickers: {failed_tickers}")

    logger.info(f"Successfully extracted {success_count}/{len(tickers)} tickers")

    return all_stock_data


def extract_single_ticker_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """
    Convenience function to extract data for a single ticker.

    :param ticker: Stock ticker symbol
    :type ticker: str
    :param start_date: Start date in YYYY-MM-DD format
    :type start_date: str, optional
    :param end_date: End date in YYYY-MM-DD format
    :type end_date: str, optional
    :return: DataFrame with Price and Returns, or None if failed
    :rtype: pd.DataFrame or None

    Example:
        >>> df = extract_single_ticker_data("AAPL", "2024-01-01")
        >>> df["Price"].iloc[-1]  # Latest price
        195.27
    """
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    return _extract_single_ticker(ticker, start_date, end_date)


__all__ = [
    "extract_data",
    "extract_single_ticker_data",
]
