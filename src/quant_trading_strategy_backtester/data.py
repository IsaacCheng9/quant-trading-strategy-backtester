"""
Contains functions to fetch financial data from external sources such as
Yahoo Finance and Wikipedia.
"""

import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import pandas as pd
import polars as pl
import requests
import streamlit as st
import yfinance as yf

from quant_trading_strategy_backtester.utils import logger


@st.cache_data
def load_yfinance_data_one_ticker(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """
    Fetches adjusted historical stock data for a ticker from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol.
        start_date: The start date for the data.
        end_date: The end date for the data.

    Returns:
        A Polars DataFrame containing adjusted historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    data = cast(pd.DataFrame, data)
    # Handle MultiIndex columns by taking just the first level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Reset index to make Date a regular column
    data = data.reset_index()

    return pl.from_pandas(data)


@st.cache_data
def load_yfinance_data_two_tickers(
    ticker1: str, ticker2: str, start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """
    Fetches adjusted historical stock data for two tickers from Yahoo Finance.

    Args:
        ticker1: The first stock ticker symbol.
        ticker2: The second stock ticker symbol.
        start_date: The start date for the data.
        end_date: The end date for the data.

    Returns:
        A Polars DataFrame containing adjusted historical stock data for both tickers.
    """
    ticker1 = ticker1.strip().upper()
    ticker2 = ticker2.strip().upper()
    if ticker1 == ticker2:
        raise ValueError("Ticker symbols must be different")

    # Download both tickers in one call for better performance.
    data = yf.download(
        [ticker1, ticker2], start=start_date, end=end_date, auto_adjust=True
    )
    data = cast(pd.DataFrame, data)
    if data.empty:
        return _empty_two_ticker_data()

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns for two-ticker yfinance data")

    close_1 = _select_close_series(data, ticker1)
    close_2 = _select_close_series(data, ticker2)
    close_data = pd.DataFrame(
        {
            "Date": data.index,
            "Close_1": close_1.to_numpy(),
            "Close_2": close_2.to_numpy(),
        }
    )

    # Convert to Polars.
    combined_data = pl.from_pandas(close_data)
    # Remove any rows with null values (different exchange calendars).
    combined_data = combined_data.drop_nulls()

    return combined_data


def _select_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Select a ticker's close series from either yfinance MultiIndex layout.

    Args:
        data: The yfinance response for multiple tickers.
        ticker: The ticker whose close prices should be selected.

    Returns:
        The ticker's close-price series.

    Raises:
        ValueError: If the response does not contain exactly one close column.
    """
    matches = [
        column
        for column in data.columns
        if isinstance(column, tuple) and "Close" in column and ticker in column
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Close column for {ticker}, found {len(matches)}"
        )

    return data[matches[0]]


def _empty_two_ticker_data() -> pl.DataFrame:
    """Return the empty schema expected by pairs-trading callers."""
    return pl.DataFrame(
        schema={
            "Date": pl.Datetime,
            "Close_1": pl.Float64,
            "Close_2": pl.Float64,
        }
    )


@st.cache_data
def get_ticker_market_cap(ticker: str) -> tuple[str, float | None]:
    """
    Fetch market cap data for a single ticker from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        A tuple containing the ticker symbol and market cap if available.
    """
    data = yf.Ticker(ticker).info
    market_cap = data.get("marketCap")
    if market_cap is None:
        logger.error(f"Market cap data for {ticker} is unavailable")
        return ticker, None

    return ticker, market_cap


@st.cache_data
def get_top_sp500_companies(num_companies: int) -> list[tuple[str, float]]:
    """
    Fetches the top X companies in the S&P 500 index by market cap.

    Args:
        num_companies: The number of top companies to fetch.

    Returns:
        A list of tuples containing the ticker symbols and market cap
        of each company in the top X of the S&P 500 index, sorted by market
        cap.
    """
    # Fetch the list of S&P 500 companies from the Wikipedia table.
    SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(SOURCE, headers=headers)
    response.raise_for_status()
    sp500_constituents = pd.read_html(response.text)[0]
    tickers = sp500_constituents["Symbol"].to_list()

    # Fetch market cap data with threading for faster execution.
    sp500_companies = []
    with ThreadPoolExecutor() as executor:
        future_to_ticker = {
            executor.submit(get_ticker_market_cap, ticker): ticker for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker, market_cap = future.result()
            if market_cap is not None:
                sp500_companies.append((ticker, market_cap))

    # Sort companies by market cap (descending) and take the top X companies.
    sorted_companies = sorted(sp500_companies, key=lambda x: x[1], reverse=True)
    top_companies = (
        sorted_companies[:num_companies] if num_companies > 0 else sorted_companies
    )

    return top_companies


@st.cache_data
def get_full_company_name(ticker: str) -> str | None:
    """
    Fetches the full company name for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        The full company name if available, otherwise None.
    """
    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get("longName", ticker)
    except Exception as e:
        logger.error(f"Failed to fetch company name for {ticker}: {e}")
        return None


@st.cache_data
def is_same_company(ticker1: str, ticker2: str) -> bool:
    """
    Determines if two tickers represent the same underlying company.

    Args:
        ticker1: The first ticker symbol.
        ticker2: The second ticker symbol.

    Returns:
        True if the tickers likely represent the same company, False otherwise.
    """
    try:
        company1 = yf.Ticker(ticker1).info.get("longName", "").lower()
        company2 = yf.Ticker(ticker2).info.get("longName", "").lower()
        return company1 == company2 and company1 != "" and company2 != ""
    except Exception as e:
        logger.error(f"Error comparing {ticker1} and {ticker2}: {e}")
        return False
