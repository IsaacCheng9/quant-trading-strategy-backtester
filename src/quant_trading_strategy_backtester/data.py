"""
Contains functions to fetch financial data from external sources such as
Yahoo Finance and Wikipedia.
"""

import datetime
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Any, cast

import pandas as pd
import polars as pl
import requests
import yfinance as yf

from quant_trading_strategy_backtester.utils import logger

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 10
MAX_MARKET_CAP_WORKERS = 8

_CACHE_LOCK = threading.Lock()
_SP500_CIK_CACHE: dict[str, str] = {}
_COMPANY_NAME_CACHE: dict[str, str] = {}


def normalise_yfinance_ticker(ticker: str) -> str:
    """Normalise a Wikipedia S&P 500 ticker symbol for Yahoo Finance.

    Strip surrounding whitespace, convert to uppercase, and replace dots
    with hyphens so share-class tickers such as BRK.B and BF.B match Yahoo
    Finance symbols (BRK-B, BF-B).

    Args:
        ticker: The raw ticker symbol.

    Returns:
        The normalised Yahoo Finance ticker symbol.
    """
    return ticker.strip().upper().replace(".", "-")


def clear_data_caches() -> None:
    """Clear in-memory S&P 500 CIK and company-name lookup caches."""
    with _CACHE_LOCK:
        _SP500_CIK_CACHE.clear()
        _COMPANY_NAME_CACHE.clear()


def _validate_date_range(start_date: datetime.date, end_date: datetime.date) -> None:
    """Validate that start_date precedes end_date.

    Args:
        start_date: Inclusive start date for historical data.
        end_date: Inclusive end date for historical data.

    Raises:
        ValueError: If start_date is on or after end_date.
    """
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")


def load_yfinance_data_one_ticker(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """Fetch adjusted historical stock data for one ticker from Yahoo Finance.

    Advance end_date by one day when calling yfinance because Yahoo Finance
    treats its end bound as exclusive, keeping end_date inclusive for callers.

    Args:
        ticker: The stock ticker symbol.
        start_date: Inclusive start date for the data.
        end_date: Inclusive end date for the data.

    Returns:
        A Polars DataFrame containing adjusted historical stock data.

    Raises:
        ValueError: If start_date is on or after end_date.
    """
    _validate_date_range(start_date, end_date)
    exclusive_end = end_date + datetime.timedelta(days=1)
    data = yf.download(ticker, start=start_date, end=exclusive_end, auto_adjust=True)
    data = cast(pd.DataFrame, data)
    # Handle MultiIndex columns by taking just the first level.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Reset index to make Date a regular column.
    data = data.reset_index()

    return pl.from_pandas(data)


def load_yfinance_data_two_tickers(
    ticker1: str, ticker2: str, start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """Fetch adjusted historical stock data for two tickers from Yahoo Finance.

    Advance end_date by one day when calling yfinance because Yahoo Finance
    treats its end bound as exclusive, keeping end_date inclusive for callers.

    Args:
        ticker1: The first stock ticker symbol.
        ticker2: The second stock ticker symbol.
        start_date: Inclusive start date for the data.
        end_date: Inclusive end date for the data.

    Returns:
        A Polars DataFrame containing adjusted historical stock data for both
        tickers.

    Raises:
        ValueError: If start_date is on or after end_date, if both tickers are
            identical, or if yfinance returns malformed columns.
    """
    _validate_date_range(start_date, end_date)
    ticker1 = ticker1.strip().upper()
    ticker2 = ticker2.strip().upper()
    if ticker1 == ticker2:
        raise ValueError("Ticker symbols must be different")

    # Download both tickers in one call for better performance.
    exclusive_end = end_date + datetime.timedelta(days=1)
    data = yf.download(
        [ticker1, ticker2], start=start_date, end=exclusive_end, auto_adjust=True
    )
    data = cast(pd.DataFrame, data)
    if data.empty:
        return _empty_two_ticker_data()

    if not isinstance(data.columns, pd.MultiIndex):
        # Treat malformed provider data as an invalid response value.
        raise ValueError(  # noqa: TRY004
            "Expected MultiIndex columns for two-ticker yfinance data"
        )

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


def _empty_many_ticker_data() -> pl.DataFrame:
    """Return the empty schema for batched multi-ticker close prices."""
    return pl.DataFrame(schema={"Date": pl.Datetime})


def load_yfinance_data_many_tickers(
    tickers: Sequence[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> pl.DataFrame:
    """Fetch adjusted close prices for many tickers in one Yahoo Finance call.

    Preserve missing observations per ticker so pairwise extraction can drop
    null rows independently for each ticker pair.

    Args:
        tickers: Ticker symbols to download in a single batch.
        start_date: Start date for the historical price data.
        end_date: End date for the historical price data.

    Returns:
        A Polars DataFrame containing a ``Date`` column and one close-price
        column per available normalised ticker symbol.

    Raises:
        ValueError: If Yahoo Finance returns malformed columns for a
            multi-ticker request or duplicate close columns for one ticker.
    """
    normalised_tickers = list(
        dict.fromkeys(
            cleaned for ticker in tickers if (cleaned := ticker.strip().upper())
        )
    )
    if not normalised_tickers:
        return _empty_many_ticker_data()

    data = yf.download(
        normalised_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )
    data = cast(pd.DataFrame, data)
    if data.empty:
        return _empty_many_ticker_data()

    close_columns: dict[str, object] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in normalised_tickers:
            matches = [
                column
                for column in data.columns
                if isinstance(column, tuple) and "Close" in column and ticker in column
            ]
            if not matches:
                logger.warning(f"No Close column returned for {ticker}; skipping")
                continue
            if len(matches) > 1:
                raise ValueError(
                    f"Expected at most one Close column for {ticker}, "
                    f"found {len(matches)}"
                )
            close_columns[ticker] = data[matches[0]].to_numpy()
    elif len(normalised_tickers) == 1 and "Close" in data.columns:
        close_columns[normalised_tickers[0]] = data["Close"].to_numpy()
    else:
        raise ValueError("Expected MultiIndex columns for multi-ticker yfinance data")

    if not close_columns:
        return _empty_many_ticker_data()

    close_frame = pd.DataFrame({"Date": data.index, **close_columns})
    return pl.from_pandas(close_frame)


def extract_pair_close_data(
    batch_data: pl.DataFrame,
    ticker1: str,
    ticker2: str,
) -> pl.DataFrame:
    """Assemble a pairwise close DataFrame from a batched price table.

    Select ``Date`` and both tickers' close columns, rename them to
    ``Close_1`` and ``Close_2``, and drop rows with null or NaN values for
    this pair only.

    Args:
        batch_data: Wide Polars DataFrame returned by
            ``load_yfinance_data_many_tickers``.
        ticker1: First ticker symbol, mapped to ``Close_1``.
        ticker2: Second ticker symbol, mapped to ``Close_2``.

    Returns:
        A Polars DataFrame with columns ``Date``, ``Close_1``, and
        ``Close_2``, or an empty pairs DataFrame if either ticker is missing.

    Raises:
        ValueError: If both ticker symbols normalise to the same value.
    """
    normalised_1 = ticker1.strip().upper()
    normalised_2 = ticker2.strip().upper()
    if normalised_1 == normalised_2:
        raise ValueError("Ticker symbols must be different")

    if (
        batch_data.is_empty()
        or "Date" not in batch_data.columns
        or normalised_1 not in batch_data.columns
        or normalised_2 not in batch_data.columns
    ):
        return _empty_two_ticker_data()

    pair_data = batch_data.select(
        [
            pl.col("Date"),
            pl.col(normalised_1).cast(pl.Float64).alias("Close_1"),
            pl.col(normalised_2).cast(pl.Float64).alias("Close_2"),
        ]
    )
    return pair_data.filter(
        pl.col("Date").is_not_null()
        & pl.col("Close_1").is_not_null()
        & pl.col("Close_2").is_not_null()
        & pl.col("Close_1").is_not_nan()
        & pl.col("Close_2").is_not_nan()
    )


def _fetch_sp500_constituents() -> pd.DataFrame:
    """Fetch and parse the Wikipedia S&P 500 constituents table."""
    response = requests.get(
        SP500_WIKIPEDIA_URL,
        headers=SP500_REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))[0]


def _normalise_cik(value: Any) -> str | None:
    """Normalise a Wikipedia CIK cell into a canonical 10-digit string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if pd.isna(value) or not value.is_integer():
            return None
        digits = str(int(value))
    elif isinstance(value, int):
        digits = str(value)
    else:
        if bool(pd.isna(value)):
            return None
        digits = str(value).strip()
        if not digits:
            return None

    return digits.zfill(10) if digits.isdigit() else digits


def _extract_sp500_ciks(sp500_constituents: pd.DataFrame) -> dict[str, str]:
    """Extract normalised ticker-to-CIK mappings from the Wikipedia table."""
    if (
        "Symbol" not in sp500_constituents.columns
        or "CIK" not in sp500_constituents.columns
    ):
        return {}

    ciks: dict[str, str] = {}
    for symbol, cik in zip(
        sp500_constituents["Symbol"],
        sp500_constituents["CIK"],
        strict=True,
    ):
        if pd.isna(symbol):
            continue
        ticker = normalise_yfinance_ticker(str(symbol))
        normalised_cik = _normalise_cik(cik)
        if ticker and normalised_cik is not None:
            ciks[ticker] = normalised_cik

    return ciks


def get_sp500_company_ciks() -> dict[str, str]:
    """Return a cached mapping of normalised S&P 500 tickers to SEC CIKs.

    Reuse the mapping already populated by ``get_top_sp500_companies`` when
    available, or fetch the Wikipedia S&P 500 table once on demand.

    Returns:
        A dictionary mapping normalised ticker symbols to canonical CIK
        strings, or an empty dictionary if the table cannot be fetched.
    """
    with _CACHE_LOCK:
        if _SP500_CIK_CACHE:
            return dict(_SP500_CIK_CACHE)

    try:
        sp500_constituents = _fetch_sp500_constituents()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to fetch S&P 500 CIK table: {exc}")
        return {}

    ciks = _extract_sp500_ciks(sp500_constituents)
    with _CACHE_LOCK:
        _SP500_CIK_CACHE.update(ciks)
        return dict(_SP500_CIK_CACHE)


def get_ticker_market_cap(ticker: str) -> tuple[str, float | None]:
    """Fetch market cap data for a single ticker from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        A tuple containing the normalised ticker symbol and market cap if
        available.
    """
    normalised_ticker = normalise_yfinance_ticker(ticker)
    try:
        data = yf.Ticker(normalised_ticker).info
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to fetch market cap for {normalised_ticker}: {exc}")
        return normalised_ticker, None

    market_cap = data.get("marketCap")
    if market_cap is None:
        logger.error(f"Market cap data for {normalised_ticker} is unavailable")
        return normalised_ticker, None

    return normalised_ticker, market_cap


def get_top_sp500_companies(num_companies: int) -> list[tuple[str, float]]:
    """Fetch the top X companies in the S&P 500 index by market cap.

    Normalise Wikipedia dot-tickers (such as BRK.B and BF.B) to Yahoo
    Finance symbols before querying market caps, and cache the Wikipedia
    CIK column for same-company pair filtering.

    Args:
        num_companies: The number of top companies to fetch.

    Returns:
        A list of tuples containing normalised ticker symbols and market caps
        of the top X S&P 500 companies, sorted descending by market cap.
    """
    sp500_constituents = _fetch_sp500_constituents()
    ciks = _extract_sp500_ciks(sp500_constituents)
    if ciks:
        with _CACHE_LOCK:
            _SP500_CIK_CACHE.update(ciks)

    tickers = list(
        dict.fromkeys(
            normalised
            for symbol in sp500_constituents["Symbol"].to_list()
            if not pd.isna(symbol)
            and (normalised := normalise_yfinance_ticker(str(symbol)))
        )
    )

    # Fetch market cap data with bounded concurrency to limit rate-limiting.
    sp500_companies = []
    with ThreadPoolExecutor(max_workers=MAX_MARKET_CAP_WORKERS) as executor:
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


def get_full_company_name(ticker: str) -> str | None:
    """Fetch the full company name for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        The full company name if available, otherwise None.
    """
    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get("longName", ticker)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to fetch company name for {ticker}: {exc}")
        return None


def _lookup_cik(company_ciks: Mapping[str, str], ticker: str) -> str | None:
    """Look up a ticker's CIK using stripped and Yahoo-normalised keys."""
    stripped = ticker.strip().upper()
    if stripped in company_ciks:
        return company_ciks[stripped]
    normalised = normalise_yfinance_ticker(stripped)
    return company_ciks.get(normalised)


def _get_cached_company_name(ticker: str) -> str | None:
    """Fetch and cache the lowercased Yahoo Finance longName for one ticker."""
    normalised = ticker.strip().upper()
    with _CACHE_LOCK:
        if normalised in _COMPANY_NAME_CACHE:
            return _COMPANY_NAME_CACHE[normalised]

    try:
        raw_name = yf.Ticker(normalised).info.get("longName", "")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error fetching company info for {normalised}: {exc}")
        return None

    company_name = str(raw_name or "").strip().lower()
    with _CACHE_LOCK:
        _COMPANY_NAME_CACHE[normalised] = company_name
    return company_name


def is_same_company(
    ticker1: str,
    ticker2: str,
    company_ciks: Mapping[str, str] | None = None,
) -> bool:
    """Determine if two tickers represent the same underlying company.

    Prefer Wikipedia SEC CIK equality when both tickers have known CIKs
    (making zero Yahoo Finance calls). Otherwise fall back to cached
    per-ticker Yahoo Finance ``longName`` lookups so N tickers trigger at
    most N metadata requests.

    Args:
        ticker1: The first ticker symbol.
        ticker2: The second ticker symbol.
        company_ciks: Optional ticker-to-CIK mapping overriding the cached
            S&P 500 CIK table.

    Returns:
        True if the tickers likely represent the same company, False
        otherwise.
    """
    if company_ciks is not None:
        ciks: Mapping[str, str] = company_ciks
    else:
        with _CACHE_LOCK:
            ciks = dict(_SP500_CIK_CACHE)

    cik1 = _lookup_cik(ciks, ticker1)
    cik2 = _lookup_cik(ciks, ticker2)
    if cik1 is not None and cik2 is not None:
        return cik1 == cik2

    company1 = _get_cached_company_name(ticker1)
    company2 = _get_cached_company_name(ticker2)
    return bool(company1 and company2 and company1 == company2)
