"""Tests for extractor module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.extractor import extract_data


class TestExtractor:
    """Test data extraction."""

    @patch("src.extractor.yf.Ticker")
    def test_extract_data(self, mock_ticker: MagicMock) -> None:
        """Test extracting historical data."""
        # Create mock data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]},
            index=dates,
        )

        # Setup mock
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_df
        mock_ticker.return_value = mock_stock

        tickers = ["MSFT", "AAPL"]
        data = extract_data(tickers, start_date="2024-01-01")

        assert isinstance(data, dict)
        assert len(data) > 0
        for ticker in tickers:
            if ticker in data:
                assert isinstance(data[ticker], pd.DataFrame)
                assert "Price" in data[ticker].columns
                assert "Returns" in data[ticker].columns
                assert data[ticker].index.name == "Date"
                # Check that index is date type
                assert all(isinstance(d, date) for d in data[ticker].index)

    @patch("src.extractor.yf.Ticker")
    def test_extract_data_with_end_date(self, mock_ticker: MagicMock) -> None:
        """Test extracting data with end_date filter."""
        # Create mock data within the date range
        dates = pd.date_range("2024-01-01", "2024-05-31", freq="D")
        mock_df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(len(dates))]}, index=dates)

        # Setup mock
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_df
        mock_ticker.return_value = mock_stock

        tickers = ["KO"]
        end_date = "2024-06-01"
        data = extract_data(tickers, start_date="2024-01-01", end_date=end_date)

        assert isinstance(data, dict)
        if tickers[0] in data:
            assert isinstance(data[tickers[0]], pd.DataFrame)
            # Check that all dates are <= end_date
            if len(data[tickers[0]]) > 0:
                assert all(
                    pd.Timestamp(d) <= pd.Timestamp(end_date) for d in data[tickers[0]].index
                )
