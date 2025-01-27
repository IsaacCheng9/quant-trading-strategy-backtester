"""
Contains pytest fixtures for tests, such as mock data.
"""

import pandas as pd
import polars as pl
import pytest
from quant_trading_strategy_backtester.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def mock_yfinance_data() -> pd.DataFrame:
    dates = pd.date_range(start="1/1/2020", end="1/31/2020")
    data = pd.DataFrame(
        {
            "Open": [100.0] * len(dates),
            "High": [110.0] * len(dates),
            "Low": [90.0] * len(dates),
            "Close": [105.0] * len(dates),
            "Adj Close": [105.0] * len(dates),
            "Volume": [1000000] * len(dates),
        },
        index=dates,
    )
    # Convert the index to a column named 'Date'
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Date"}, inplace=True)
    return data


@pytest.fixture
def mock_polars_data(mock_yfinance_data: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(mock_yfinance_data)


@pytest.fixture
def mock_yfinance_pairs_data() -> pd.DataFrame:
    dates = pd.date_range(start="1/1/2020", end="1/31/2020")
    data = pd.DataFrame(
        {
            "Close_1": [100.0 + i * 0.1 for i in range(len(dates))],
            "Close_2": [100.0 + i * 0.05 for i in range(len(dates))],
        },
        index=dates,
    )
    # Convert the index to a column named 'Date'
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Date"}, inplace=True)
    return data


@pytest.fixture
def mock_polars_pairs_data(mock_yfinance_pairs_data: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(mock_yfinance_pairs_data)


@pytest.fixture(scope="function", autouse=True)
def mock_db_session(monkeypatch):
    def mock_session():
        return session

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.models.Session", mock_session
    )
    yield session
    session.close()


@pytest.fixture(autouse=True)
def mock_yfinance_functions(monkeypatch):
    def mock_load_one_ticker(*args, **kwargs):
        dates = pd.date_range(start="1/1/2020", end="1/31/2020")
        return pl.DataFrame(
            {"Date": dates, "Close": [100.0 + i for i in range(len(dates))]}
        )

    def mock_load_two_tickers(*args, **kwargs):
        dates = pd.date_range(start="1/1/2020", end="1/31/2020")
        return pl.DataFrame(
            {
                "Date": dates,
                "Close_1": [100.0 + i * 0.1 for i in range(len(dates))],
                "Close_2": [100.0 + i * 0.05 for i in range(len(dates))],
            }
        )

    # Mock all potential yfinance data loading functions
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.data.load_yfinance_data_one_ticker",
        mock_load_one_ticker,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.data.load_yfinance_data_two_tickers",
        mock_load_two_tickers,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        mock_load_one_ticker,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_two_tickers",
        mock_load_two_tickers,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        mock_load_one_ticker,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_two_tickers",
        mock_load_two_tickers,
    )


@pytest.fixture(autouse=True)
def mock_yfinance_download(monkeypatch):
    def mock_download(*args, **kwargs):
        return pd.DataFrame(
            {
                "Date": pd.date_range(start="2020-01-01", end="2020-12-31"),
                "Close": [100 + i * 0.1 for i in range(365)],
                "Adj Close": [100 + i * 0.1 for i in range(365)],
                "Volume": [1000000 for _ in range(365)],
            }
        ).set_index("Date")

    # Mock all potential network-related functions
    monkeypatch.setattr("yfinance.download", mock_download)
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda x: type("MockTicker", (), {"history": lambda **kwargs: mock_download()}),
    )
