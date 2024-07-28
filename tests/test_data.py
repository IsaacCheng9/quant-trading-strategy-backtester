"""
Contains tests for data fetching functions.
"""

import datetime

import pandas as pd
import polars as pl
from quant_trading_strategy_backtester.data import (
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
