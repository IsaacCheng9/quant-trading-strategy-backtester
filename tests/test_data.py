"""
Contains tests for data fetching functions.
"""

import datetime
import subprocess
import sys
from typing import Any

import pandas as pd
import polars as pl
import pytest

from quant_trading_strategy_backtester import data as data_module
from quant_trading_strategy_backtester.data import (
    MAX_MARKET_CAP_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
    extract_pair_close_data,
    get_full_company_name,
    get_sp500_company_ciks,
    get_top_sp500_companies,
    is_same_company,
    load_yfinance_data_many_tickers,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
    normalise_yfinance_ticker,
)


def test_load_yfinance_data_one_ticker(
    monkeypatch, mock_yfinance_data: pd.DataFrame
) -> None:
    def mock_download(*args, **kwargs):
        assert kwargs["auto_adjust"] is True
        assert kwargs["start"] == datetime.date(2020, 1, 1)
        assert kwargs["end"] == datetime.date(2020, 2, 1)
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
        assert kwargs["start"] == datetime.date(2020, 1, 1)
        assert kwargs["end"] == datetime.date(2020, 2, 1)
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


@pytest.mark.parametrize(
    "start_date,end_date",
    [
        (datetime.date(2020, 1, 15), datetime.date(2020, 1, 15)),
        (datetime.date(2020, 1, 16), datetime.date(2020, 1, 15)),
    ],
)
def test_load_yfinance_data_rejects_invalid_date_ranges(
    monkeypatch,
    start_date: datetime.date,
    end_date: datetime.date,
) -> None:
    """Verify both yfinance loaders reject start_date >= end_date up front."""
    monkeypatch.setattr(
        "yfinance.download",
        lambda *_args, **_kwargs: pytest.fail("yf.download should not be called"),
    )

    with pytest.raises(ValueError, match="start_date must be earlier than end_date"):
        load_yfinance_data_one_ticker("AAPL", start_date, end_date)

    with pytest.raises(ValueError, match="start_date must be earlier than end_date"):
        load_yfinance_data_two_tickers("AAPL", "MSFT", start_date, end_date)


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


def test_normalise_yfinance_ticker_converts_wikipedia_dot_tickers() -> None:
    """Verify Wikipedia share-class dots normalise to Yahoo Finance hyphens."""
    assert normalise_yfinance_ticker("BRK.B") == "BRK-B"
    assert normalise_yfinance_ticker(" bf.b ") == "BF-B"
    assert normalise_yfinance_ticker("aapl") == "AAPL"
    assert normalise_yfinance_ticker("BRK-B") == "BRK-B"


def test_get_top_sp500_companies_normalises_tickers_caps_workers_and_caches_ciks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Wikipedia fetch uses timeout, worker cap, dot norm, and CIKs."""
    html = """
    <table>
      <thead>
        <tr><th>Symbol</th><th>Security</th><th>CIK</th></tr>
      </thead>
      <tbody>
        <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>0001067983</td></tr>
        <tr><td>BF.B</td><td>Brown-Forman</td><td>0000014693</td></tr>
        <tr><td>GOOG</td><td>Alphabet Inc. (Class C)</td><td>0001652044</td></tr>
        <tr><td>GOOGL</td><td>Alphabet Inc. (Class A)</td><td>0001652044</td></tr>
        <tr><td>MSFT</td><td>Microsoft</td><td>0000789019</td></tr>
      </tbody>
    </table>
    """
    captured_requests: list[dict[str, object]] = []
    captured_max_workers: list[int | None] = []
    queried_tickers: list[str] = []

    class FakeResponse:
        text = html

        def raise_for_status(self) -> None:
            return None

    def mock_requests_get(url: str, **kwargs: object) -> FakeResponse:
        captured_requests.append({"url": url, **kwargs})
        return FakeResponse()

    real_executor = data_module.ThreadPoolExecutor

    class TrackingExecutor(real_executor):
        def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
            captured_max_workers.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    market_caps = {
        "BRK-B": 900.0,
        "BF-B": 100.0,
        "GOOG": 2000.0,
        "GOOGL": 1900.0,
        "MSFT": 3000.0,
    }

    def mock_ticker(symbol: str):
        queried_tickers.append(symbol)

        class MockTicker:
            @property
            def info(self) -> dict[str, object]:
                return {"marketCap": market_caps[symbol]}

        return MockTicker()

    monkeypatch.setattr("requests.get", mock_requests_get)
    monkeypatch.setattr(data_module, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr("yfinance.Ticker", mock_ticker)

    top_companies = get_top_sp500_companies(4)

    assert len(captured_requests) == 1
    assert captured_requests[0]["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert captured_max_workers == [MAX_MARKET_CAP_WORKERS]
    assert set(queried_tickers) == {"BRK-B", "BF-B", "GOOG", "GOOGL", "MSFT"}
    assert top_companies == [
        ("MSFT", 3000.0),
        ("GOOG", 2000.0),
        ("GOOGL", 1900.0),
        ("BRK-B", 900.0),
    ]

    # Subsequent same-company checks must use cached CIKs (zero Yahoo calls).
    queried_tickers.clear()
    assert is_same_company("GOOG", "GOOGL") is True
    assert is_same_company("BRK.B", "MSFT") is False
    assert is_same_company("GOOG", "MSFT") is False
    assert queried_tickers == []


def test_get_sp500_company_ciks_fetches_once_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_sp500_company_ciks caches Wikipedia CIK mappings."""
    html = """
    <table>
      <thead>
        <tr><th>Symbol</th><th>CIK</th></tr>
      </thead>
      <tbody>
        <tr><td>FOX</td><td>1754301</td></tr>
        <tr><td>FOXA</td><td>0001754301</td></tr>
        <tr><td>NWS</td><td>1564708</td></tr>
        <tr><td>NWSA</td><td>1564708</td></tr>
        <tr><td>BRK.B</td><td>1067983</td></tr>
      </tbody>
    </table>
    """
    request_calls: list[dict[str, object]] = []

    class FakeResponse:
        text = html

        def raise_for_status(self) -> None:
            return None

    def mock_requests_get(url: str, **kwargs: object) -> FakeResponse:
        request_calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr("requests.get", mock_requests_get)

    first_ciks = get_sp500_company_ciks()
    second_ciks = get_sp500_company_ciks()

    assert len(request_calls) == 1
    assert request_calls[0]["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert first_ciks == second_ciks
    assert first_ciks["FOX"] == first_ciks["FOXA"] == "0001754301"
    assert first_ciks["NWS"] == first_ciks["NWSA"] == "0001564708"
    assert first_ciks["BRK-B"] == "0001067983"


def test_is_same_company_uses_explicit_ciks_without_yfinance_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit company_ciks avoids all Yahoo Finance .info lookups."""
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda *_args, **_kwargs: pytest.fail("Yahoo Finance should not be called"),
    )
    ciks = {
        "FOX": "0001754301",
        "FOXA": "0001754301",
        "NWS": "0001564708",
        "NWSA": "0001564708",
        "AAPL": "0000320193",
    }

    assert is_same_company("FOX", "FOXA", company_ciks=ciks) is True
    assert is_same_company("NWS", "NWSA", company_ciks=ciks) is True
    assert is_same_company("FOX", "AAPL", company_ciks=ciks) is False


def test_is_same_company_caches_per_ticker_fallback_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback name lookups call Yahoo at most once per ticker."""
    names = {
        "GOOG": "Alphabet Inc.",
        "GOOGL": "Alphabet Inc.",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
    }
    info_calls: list[str] = []

    def mock_ticker(symbol: str):
        class MockTicker:
            @property
            def info(self) -> dict[str, str]:
                info_calls.append(symbol)
                return {"longName": names[symbol]}

        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker)

    tickers = ["GOOG", "GOOGL", "AAPL", "MSFT"]
    matches = [
        (t1, t2)
        for idx, t1 in enumerate(tickers)
        for t2 in tickers[idx + 1 :]
        if is_same_company(t1, t2)
    ]

    assert matches == [("GOOG", "GOOGL")]
    # 6 pairs evaluated, but at most N = 4 Yahoo .info lookups.
    assert len(info_calls) == 4
    assert set(info_calls) == set(tickers)


def test_load_yfinance_data_many_tickers_and_extract_pair_close_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batch download fetches once and drops nulls per pair only."""
    dates = pd.date_range(start="2020-01-01", periods=4)
    download_calls: list[list[str]] = []

    def mock_download(tickers, **kwargs):
        download_calls.append(list(tickers))
        assert kwargs["auto_adjust"] is True
        columns = pd.MultiIndex.from_tuples(
            [
                ("Close", "AAPL"),
                ("Close", "MSFT"),
                ("Close", "GOOGL"),
                ("Volume", "AAPL"),
            ]
        )
        return pd.DataFrame(
            [
                [100.0, 200.0, float("nan"), 1000.0],
                [101.0, 201.0, 301.0, 1100.0],
                [102.0, float("nan"), 302.0, 1200.0],
                [103.0, 203.0, 303.0, 1300.0],
            ],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", mock_download)

    batch = load_yfinance_data_many_tickers(
        ["aapl", " MSFT ", "GOOGL", "AAPL", "MISSING"],
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 5),
    )

    assert download_calls == [["AAPL", "MSFT", "GOOGL", "MISSING"]]
    assert batch.columns == ["Date", "AAPL", "MSFT", "GOOGL"]
    assert len(batch) == 4

    aapl_msft = extract_pair_close_data(batch, "AAPL", "MSFT")
    assert aapl_msft.columns == ["Date", "Close_1", "Close_2"]
    assert aapl_msft["Close_1"].to_list() == [100.0, 101.0, 103.0]
    assert aapl_msft["Close_2"].to_list() == [200.0, 201.0, 203.0]

    aapl_googl = extract_pair_close_data(batch, "AAPL", "GOOGL")
    assert aapl_googl["Close_1"].to_list() == [101.0, 102.0, 103.0]
    assert aapl_googl["Close_2"].to_list() == [301.0, 302.0, 303.0]

    missing_pair = extract_pair_close_data(batch, "AAPL", "MISSING")
    assert missing_pair.is_empty()
    assert missing_pair.columns == ["Date", "Close_1", "Close_2"]


def test_load_yfinance_data_many_tickers_handles_ticker_first_and_single_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batch download handles ticker-first MultiIndex and flat data."""
    dates = pd.date_range(start="2020-01-01", periods=3)

    def mock_ticker_first(tickers, **kwargs):
        columns = pd.MultiIndex.from_tuples([("AAPL", "Close"), ("MSFT", "Close")])
        return pd.DataFrame(
            [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", mock_ticker_first)
    batch = load_yfinance_data_many_tickers(
        ["AAPL", "MSFT"],
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 4),
    )
    assert batch["AAPL"].to_list() == [10.0, 11.0, 12.0]
    assert batch["MSFT"].to_list() == [20.0, 21.0, 22.0]

    def mock_single_flat(tickers, **kwargs):
        return pd.DataFrame({"Close": [50.0, 51.0, 52.0]}, index=dates)

    monkeypatch.setattr("yfinance.download", mock_single_flat)
    single_batch = load_yfinance_data_many_tickers(
        ["AAPL"],
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 4),
    )
    assert single_batch.columns == ["Date", "AAPL"]
    assert single_batch["AAPL"].to_list() == [50.0, 51.0, 52.0]
