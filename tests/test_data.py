"""
Contains tests for data fetching functions.
"""
import datetime

import pandas as pd
from quant_trading_strategy_backtester.data import (
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)


def test_load_yfinance_data_one_ticker(monkeypatch, mock_data: pd.DataFrame) -> None:
    def mock_download(*args, **kwargs):
        return mock_data

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_one_ticker(
        "AAPL", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "Close" in data.columns


def test_load_yfinance_data_two_tickers(monkeypatch, mock_data: pd.DataFrame) -> None:
    def mock_download(*args, **kwargs):
        return mock_data

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_two_tickers(
        "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "Close_1" in data.columns
    assert "Close_2" in data.columns
