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
