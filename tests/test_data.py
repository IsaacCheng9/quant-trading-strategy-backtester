"""
Contains tests for data fetching functions.
"""

import datetime
import subprocess
import sys

import pandas as pd
import polars as pl
import pytest

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
        assert kwargs["auto_adjust"] is True
        data = mock_yfinance_data.copy()
        data["Close"] = data["Adj Close"]
        return data.drop(columns=["Adj Close"]).set_index("Date")

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_one_ticker(
        "AAPL", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pl.DataFrame)
    assert not data.is_empty()
    assert "Date" in data.columns
    assert "Close" in data.columns
    assert "Adj Close" not in data.columns
    assert len(data) == 31


def test_load_yfinance_data_two_tickers(
    monkeypatch, mock_yfinance_data: pd.DataFrame
) -> None:
    def mock_download(*args, **kwargs):
        assert kwargs["auto_adjust"] is True
        # yfinance returns MultiIndex columns when downloading multiple
        # tickers.
        base_data = mock_yfinance_data.set_index("Date")
        multi_index_data = pd.DataFrame(
            {
                ("Close", "AAPL"): base_data["Close"],
                ("Close", "MSFT"): base_data["Close"],
                ("Open", "AAPL"): base_data["Open"],
                ("Open", "MSFT"): base_data["Open"],
                ("High", "AAPL"): base_data["High"],
                ("High", "MSFT"): base_data["High"],
            }
        )
        # Pandas stubs lose tuple-key column types during DataFrame construction.
        multi_index_data.columns = pd.MultiIndex.from_tuples(
            list(multi_index_data.columns)  # pyrefly: ignore[bad-argument-type]
        )
        return multi_index_data

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


def test_load_yfinance_data_two_tickers_preserves_requested_order(
    monkeypatch,
) -> None:
    dates = pd.date_range(start="1/1/2020", end="1/3/2020")

    def mock_download(*args, **kwargs):
        assert args[0] == ["AAPL", "MSFT"]
        columns = pd.MultiIndex.from_tuples([("Close", "MSFT"), ("Close", "AAPL")])
        return pd.DataFrame(
            [[200.0, 100.0], [201.0, 101.0], [202.0, 102.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_two_tickers(
        "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)
    )

    assert data["Close_1"].to_list() == [100.0, 101.0, 102.0]
    assert data["Close_2"].to_list() == [200.0, 201.0, 202.0]


def test_load_yfinance_data_two_tickers_handles_ticker_first_columns(
    monkeypatch,
) -> None:
    dates = pd.date_range(start="1/1/2020", end="1/3/2020")

    def mock_download(*args, **kwargs):
        columns = pd.MultiIndex.from_tuples([("AAPL", "Close"), ("MSFT", "Close")])
        return pd.DataFrame(
            [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_two_tickers(
        "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)
    )

    assert data["Close_1"].to_list() == [100.0, 101.0, 102.0]
    assert data["Close_2"].to_list() == [200.0, 201.0, 202.0]


def test_load_yfinance_data_two_tickers_rejects_flat_response(
    monkeypatch,
) -> None:
    dates = pd.date_range(start="1/1/2020", end="1/3/2020")

    def mock_download(*args, **kwargs):
        return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)

    monkeypatch.setattr("yfinance.download", mock_download)

    with pytest.raises(ValueError, match="Expected MultiIndex columns"):
        load_yfinance_data_two_tickers(
            "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)
        )


def test_load_yfinance_data_two_tickers_requires_both_close_columns(
    monkeypatch,
) -> None:
    dates = pd.date_range(start="1/1/2020", end="1/3/2020")

    def mock_download(*args, **kwargs):
        columns = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Open", "MSFT")])
        return pd.DataFrame(
            [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", mock_download)

    with pytest.raises(ValueError, match="Expected exactly one Close column for MSFT"):
        load_yfinance_data_two_tickers(
            "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)
        )


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
        raise RuntimeError("API Error")

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
        raise RuntimeError("API Error")

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_info_error)
    assert is_same_company("ERROR1", "ERROR2") is False


def test_importing_data_does_not_import_streamlit_or_warn() -> None:
    """Verify importing data outside Streamlit does not load it or warn."""
    script = (
        "import logging, sys, warnings\n"
        "log_records = []\n"
        "class CaptureHandler(logging.Handler):\n"
        "    def emit(self, record):\n"
        "        if record.levelno >= logging.WARNING:\n"
        "            log_records.append(record.getMessage())\n"
        "logging.getLogger().addHandler(CaptureHandler())\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import quant_trading_strategy_backtester.data\n"
        "assert 'streamlit' not in sys.modules, 'streamlit was imported'\n"
        "assert not caught, f'Unexpected warnings: {caught}'\n"
        "assert not log_records, f'Unexpected log warnings: {log_records}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "No runtime found" not in result.stderr
    assert "streamlit" not in result.stderr.lower()
