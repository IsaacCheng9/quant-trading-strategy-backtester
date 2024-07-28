"""
Contains pytest fixtures for tests, such as mock data.
"""

import pandas as pd
import polars as pl
import pytest


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
            "Close": [100.0 + i * 0.1 for i in range(len(dates))],
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
