"""
Contains tests for data fetching functions.
"""

import datetime
import pytest

import pandas as pd
import polars as pl
from quant_trading_strategy_backtester.data import (
    get_full_company_name,
    is_same_company,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)


def test_load_yfinance_data_one_ticker(
    monkeypatch, mock_yfinance_data: pd.DataFrame
) -> None:
    def mock_download(*args, **kwargs):
        return mock_yfinance_data.set_index("Date")

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_one_ticker(
        "AAPL", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pl.DataFrame)
    assert not data.is_empty()
    assert "Date" in data.columns
    assert "Close" in data.columns
    assert len(data) == 31


def test_load_yfinance_data_two_tickers(
    monkeypatch, mock_yfinance_data: pd.DataFrame
) -> None:
    def mock_download(*args, **kwargs):
        return mock_yfinance_data.set_index("Date")

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_two_tickers(
        "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pl.DataFrame)
    assert not data.is_empty()
    assert "Date" in data.columns
    assert "Close_1" in data.columns
    assert "Close_2" in data.columns
    assert len(data) == 31


def test_get_full_company_name_success(monkeypatch):
    def mock_ticker_info(*args, **kwargs):
        class MockTicker:
            @property
            def info(self):
                return {"longName": "Apple Inc."}

        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info)

    # Test successful retrieval
    assert get_full_company_name("AAPL") == "Apple Inc."


def test_get_full_company_name_failure(monkeypatch):
    # Test fallback to ticker when longName is not available
    def mock_ticker_info_no_long_name(*args, **kwargs):
        class MockTicker:
            @property
            def info(self):
                return {}

        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info_no_long_name)
    assert get_full_company_name("UNKNOWN") == "UNKNOWN"

    # Test error handling
    def mock_ticker_info_error(*args, **kwargs):
        raise Exception("API Error")

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info_error)
    assert get_full_company_name("ERROR") is None


def test_is_same_company_check_success(monkeypatch):
    def mock_ticker_info(*args, **kwargs):
        class MockTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            @property
            def info(self):
                if self.ticker == "GOOGL" or self.ticker == "GOOG":
                    return {"longName": "Alphabet Inc."}
                elif self.ticker == "AAPL":
                    return {"longName": "Apple Inc."}
                else:
                    return {}

        return MockTicker(args[0])

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info)

    # Test same company
    assert is_same_company("GOOGL", "GOOG") is True
    # Test different companies
    assert is_same_company("GOOGL", "AAPL") is False
    # Test with missing info
    assert is_same_company("UNKNOWN1", "UNKNOWN2") is False


def test_is_same_company_check_failure(monkeypatch):
    # Test error handling
    def mock_ticker_info_error(*args, **kwargs):
        raise Exception("API Error")

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info_error)
    assert is_same_company("ERROR1", "ERROR2") is False
