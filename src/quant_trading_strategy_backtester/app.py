"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""

import datetime
import json
import math
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import polars as pl
import streamlit as st

from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.benchmark import (
    BENCHMARK_TICKER,
    calculate_benchmark_relative_metrics,
)
from quant_trading_strategy_backtester.cointegration import evaluate_cointegration
from quant_trading_strategy_backtester.data import (
    get_full_company_name,
    get_top_sp500_companies,
    is_same_company,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)
from quant_trading_strategy_backtester.optimiser import (
    ProgressCallback,
    create_strategy,
    evaluate_validation_period,
    get_training_data,
    get_validation_data,
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    optimise_single_ticker_strategy_ticker,
    run_backtest,
    run_optimisation,
)
from quant_trading_strategy_backtester.storage import (
    InMemoryStrategyStore,
    SQLiteStrategyStore,
    StrategyStore,
)
from quant_trading_strategy_backtester.streamlit_ui import (
    get_user_inputs_except_strategy_params,
    get_user_inputs_for_strategy_params,
)
from quant_trading_strategy_backtester.utils import (
    NUM_TOP_COMPANIES_ONE_TICKER,
    NUM_TOP_COMPANIES_TWO_TICKERS,
)
from quant_trading_strategy_backtester.visualisation import (
    display_benchmark_metrics,
    display_performance_metrics,
    display_returns_by_month,
    display_trade_ledger,
    display_walk_forward_results,
    plot_equity_curve,
    plot_pairs_spread,
    plot_strategy_returns,
)


@st.cache_data
def _cached_load_yfinance_data_one_ticker(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """Cache single-ticker Yahoo Finance downloads across Streamlit reruns."""
    return load_yfinance_data_one_ticker(ticker, start_date, end_date)


@st.cache_data
def _cached_load_yfinance_data_two_tickers(
    ticker1: str,
    ticker2: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pl.DataFrame:
    """Cache two-ticker Yahoo Finance downloads across Streamlit reruns."""
    return load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)


@st.cache_data
def _cached_get_top_sp500_companies(
    num_companies: int,
) -> list[tuple[str, float]]:
    """Cache top S&P 500 market-cap rankings across Streamlit reruns."""
    return get_top_sp500_companies(num_companies)


@st.cache_data
def _cached_get_full_company_name(ticker: str) -> str | None:
    """Cache Yahoo Finance company name lookups across Streamlit reruns."""
    return get_full_company_name(ticker)


@st.cache_data
def _cached_is_same_company(ticker1: str, ticker2: str) -> bool:
    """Cache Yahoo Finance same-company checks across Streamlit reruns."""
    return is_same_company(ticker1, ticker2)


@contextmanager
def _streamlit_progress() -> Generator[ProgressCallback]:
    """Yield a progress callback backed by Streamlit progress widgets."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update(current: int, total: int, message: str) -> None:
        status_text.text(message)
        if total > 0:
            progress_bar.progress(min(max(current / total, 0.0), 1.0))

    try:
        yield update
    finally:
        progress_bar.empty()
        status_text.empty()


def _display_status_message(message: str) -> None:
    """Display an informational optimisation status message in Streamlit."""
    st.info(message)


def _run_optimisation_with_ui(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    start_date: datetime.date,
    end_date: datetime.date,
    tickers: str | list[str],
    walk_forward: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run optimisation with Streamlit progress and summary widgets."""
    st.info("Optimising strategy. This may take a while...")
    start_time = time.time()

    with _streamlit_progress() as progress_callback:
        optimised_params, metrics = run_optimisation(
            data,
            strategy_type,
            strategy_params,
            start_date,
            end_date,
            tickers,
            walk_forward=walk_forward,
            progress_callback=progress_callback,
            status_callback=_display_status_message,
            fold_results_callback=display_walk_forward_results,
            load_one_ticker=_cached_load_yfinance_data_one_ticker,
            get_top_companies=_cached_get_top_sp500_companies,
        )

    duration = time.time() - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")
    st.header("Optimal Parameters")
    st.write(optimised_params)
    return optimised_params, metrics


@st.cache_data
def _cached_run_backtest(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    tickers: str | list[str],
) -> tuple[pl.DataFrame, dict]:
    """
    Cached wrapper for run_backtest to avoid recomputation on
    Streamlit reruns.
    """
    return run_backtest(data, strategy_type, strategy_params, tickers)


@st.cache_data
def _cached_evaluate_validation_period(
    context_data: pl.DataFrame,
    validation_data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    tickers: str | list[str],
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Cache context-aware validation evaluation across Streamlit reruns."""
    return evaluate_validation_period(
        context_data,
        validation_data,
        strategy_type,
        strategy_params,
        tickers,
    )


def _get_final_backtest_data(
    data: pl.DataFrame, use_validation_data: bool, walk_forward: bool
) -> pl.DataFrame:
    """
    Return the data used for the displayed and saved final backtest.

    Args:
        data: Full historical price data.
        use_validation_data: Whether ticker selection or parameter optimisation
            has already used the training period.
        walk_forward: Whether parameter optimisation used walk-forward
            validation.

    Returns:
        Full data for plain backtests, or held-out validation data for
        optimised backtests.
    """
    if not use_validation_data:
        st.info(
            f"Displaying full-period backtest results over {len(data)} rows. "
            "No ticker-selection or parameter-optimisation split was applied."
        )
        return data

    validation_data = get_validation_data(data, walk_forward=walk_forward)
    training_data = get_training_data(data, walk_forward=walk_forward)
    if walk_forward:
        st.info(
            "Displaying final walk-forward test-fold backtest results "
            f"over {len(validation_data)} rows after expanding-window "
            f"optimisation on {len(training_data)} prior rows."
        )
    else:
        st.info(
            "Displaying held-out test-period backtest results over "
            f"{len(validation_data)} rows after selection or optimisation "
            f"on {len(training_data)} training rows."
        )
    return validation_data


def _display_cointegration_result(data: pl.DataFrame, context: str) -> None:
    """
    Display an Engle-Granger cointegration diagnostic for a selected pair.

    Args:
        data: Pair price data to test.
        context: Description of the data period being tested.
    """
    result = evaluate_cointegration(data)
    if result.is_cointegrated:
        st.success(
            "Cointegration filter passed "
            f"({context}; Engle-Granger p-value: {result.p_value:.4f})."
        )
        return

    reason = f" {result.reason}." if result.reason else ""
    st.warning(
        "Cointegration filter failed "
        f"({context}; Engle-Granger p-value: {_format_p_value(result.p_value)})."
        f"{reason} Manual pairs are still allowed, but the relationship may "
        "not be mean-reverting."
    )


def _format_p_value(value: float) -> str:
    """Format p-values for Streamlit diagnostics."""
    if math.isnan(value):
        return "N/A"

    return f"{value:.4f}"


def _get_prior_date(
    full_data: pl.DataFrame,
    backtest_data: pl.DataFrame,
) -> datetime.date | datetime.datetime | None:
    """Return the last trading date strictly before the backtest period.

    Args:
        full_data: Full historical price data containing a Date column.
        backtest_data: Displayed backtest slice containing a Date column.

    Returns:
        The latest Date in full_data strictly before backtest_data's first
        Date, or None when backtest_data starts at the beginning of full_data.
    """
    if (
        full_data.is_empty()
        or backtest_data.is_empty()
        or "Date" not in full_data.columns
        or "Date" not in backtest_data.columns
    ):
        return None

    backtest_start = backtest_data.select(pl.col("Date").min()).item()
    if backtest_start is None:
        return None

    prior_dates = full_data.filter(
        pl.col("Date") < pl.lit(backtest_start, dtype=full_data["Date"].dtype)
    )
    if prior_dates.is_empty():
        return None

    prior_date = prior_dates.select(pl.col("Date").max()).item()
    return cast(datetime.date | datetime.datetime | None, prior_date)


def _get_benchmark_date_range(
    backtest_data: pl.DataFrame,
    prior_date: datetime.date | datetime.datetime | None = None,
) -> tuple[datetime.date, datetime.date]:
    """Return inclusive start and exclusive end dates for benchmark loading.

    Args:
        backtest_data: Backtest data containing a Date column.
        prior_date: Optional trading date immediately preceding backtest_data
            to include when computing day-0 benchmark returns.

    Returns:
        A start date and Yahoo-compatible exclusive end date.
    """
    backtest_start = _to_date(backtest_data["Date"].min())
    start_date = (
        min(_to_date(prior_date), backtest_start)
        if prior_date is not None
        else backtest_start
    )
    end_date = _to_date(backtest_data["Date"].max()) + datetime.timedelta(days=1)
    return start_date, end_date


def _to_date(value: Any) -> datetime.date:
    """Convert Polars scalar date values to Python dates."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if hasattr(value, "date"):
        date_value = value.date()
        if isinstance(date_value, datetime.date):
            return date_value

    raise TypeError(f"Expected date-like value, got {type(value).__name__}")


def _display_benchmark_comparison(
    results: pl.DataFrame,
    backtest_data: pl.DataFrame,
    full_data: pl.DataFrame | None = None,
) -> None:
    """Display strategy metrics relative to the default benchmark.

    Args:
        results: Backtest results for the displayed period.
        backtest_data: Price data for the displayed period.
        full_data: Optional full historical price data used to locate the
            trading date immediately before a validation slice.
    """
    prior_date = (
        _get_prior_date(full_data, backtest_data) if full_data is not None else None
    )
    try:
        benchmark_start_date, benchmark_end_date = _get_benchmark_date_range(
            backtest_data, prior_date=prior_date
        )
        benchmark_data = _cached_load_yfinance_data_one_ticker(
            BENCHMARK_TICKER, benchmark_start_date, benchmark_end_date
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Benchmark comparison unavailable: {exc}")
        return

    benchmark_metrics = calculate_benchmark_relative_metrics(
        results, benchmark_data, prior_date=prior_date
    )
    if benchmark_metrics["Benchmark Observations"] == 0:
        st.warning(
            "Benchmark comparison unavailable: no overlapping benchmark data "
            f"for {BENCHMARK_TICKER}."
        )
        return

    display_benchmark_metrics(benchmark_metrics, BENCHMARK_TICKER)


# Trading strategy preparation functions


def _validate_automatic_selection_mode(walk_forward: bool) -> None:
    """Reject walk-forward reporting after automatic ticker selection."""
    if walk_forward:
        raise ValueError(
            "Walk-forward validation cannot be combined with automatic ticker "
            "selection because the selection window overlaps early test folds"
        )


def _display_current_universe_warning() -> None:
    """Warn that automatic selection does not use point-in-time universes."""
    st.warning(
        "Exploratory selection only: candidates use today's S&P 500 "
        "constituents and current market caps. Historical results therefore "
        "contain survivorship and future-information bias and are not "
        "point-in-time evidence."
    )


def prepare_buy_and_hold_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the optimisation process for Buy and Hold strategy.

    Selects the best ticker from the top S&P 500 companies.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The selected ticker symbol.
            - An empty dictionary (no strategy parameters for Buy and Hold).
    """
    _display_current_universe_warning()
    st.info(
        f"Ranking today's top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 "
        "constituents. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies.
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = _cached_get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)

    # Optimise ticker selection.
    with _streamlit_progress() as progress_callback:
        best_ticker, _, _ = optimise_buy_and_hold_ticker(
            top_companies,
            start_date,
            end_date,
            progress_callback=progress_callback,
            status_callback=_display_status_message,
            load_one_ticker=_cached_load_yfinance_data_one_ticker,
        )

    # Calculate and display the time taken for optimisation.
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    st.header("Exploratory Current-Universe Result")
    st.write(f"Selected ticker: {best_ticker}")

    # Load historical data for the selected ticker.
    data = _cached_load_yfinance_data_one_ticker(best_ticker, start_date, end_date)

    return data, best_ticker, {}


def prepare_single_ticker_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
    optimise: bool,
    walk_forward: bool = False,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the optimisation process for single ticker strategies.

    Selects the best ticker from the top S&P 500 companies and optimises
    strategy parameters if requested.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.
        walk_forward: Whether to use walk-forward validation.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The selected ticker symbol.
            - Optimised strategy parameters.
    """
    _validate_automatic_selection_mode(walk_forward)
    _display_current_universe_warning()
    st.info(
        f"Ranking today's top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 "
        "constituents. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies.
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = _cached_get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)

    # Optimise ticker selection.
    with _streamlit_progress() as progress_callback:
        best_ticker = optimise_single_ticker_strategy_ticker(
            top_companies,
            start_date,
            end_date,
            strategy_type,
            strategy_params,
            progress_callback=progress_callback,
            status_callback=_display_status_message,
            load_one_ticker=_cached_load_yfinance_data_one_ticker,
        )

    # Load historical data for the selected ticker.
    data = _cached_load_yfinance_data_one_ticker(best_ticker, start_date, end_date)

    # Optimise strategy parameters if requested.
    if optimise:
        if not walk_forward:
            st.warning(
                "Parameter optimisation without walk-forward validation "
                "uses a single 70/30 train/test split. Results may still "
                "be partially in-sample. Enable walk-forward validation "
                "for more robust out-of-sample evaluation."
            )
        best_params, _ = _run_optimisation_with_ui(
            data,
            strategy_type,
            strategy_params,
            start_date,
            end_date,
            best_ticker,
            walk_forward=walk_forward,
        )
    else:
        best_params = {
            k: v[0] if isinstance(v, (list, range)) else v
            for k, v in strategy_params.items()
        }

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    st.header("Exploratory Current-Universe Result")
    if optimise:
        result = {
            "ticker": best_ticker,
            **best_params,
        }
    else:
        result = {
            "ticker": best_ticker,
        }
    st.write(result)

    return data, best_ticker, best_params


def prepare_pairs_trading_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
    walk_forward: bool = False,
) -> tuple[pl.DataFrame, str, dict[str, int | float]]:
    """
    Handles the optimisation process for pairs trading strategy.

    Selects the best pair of tickers from the top S&P 500 companies and
    optimises the strategy parameters if requested.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.
        walk_forward: Whether to use walk-forward validation.

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters.
    """
    _validate_automatic_selection_mode(walk_forward)
    _display_current_universe_warning()
    # Inform the user that the optimisation process is starting
    st.info(
        f"Ranking pairs from today's top {NUM_TOP_COMPANIES_TWO_TICKERS} S&P 500 "
        "constituents. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies.
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = _cached_get_top_sp500_companies(NUM_TOP_COMPANIES_TWO_TICKERS)

    parameter_ranges = strategy_params

    # Optimise ticker pair selection and strategy parameters.
    with _streamlit_progress() as progress_callback:
        ticker, selected_strategy_params, _ = optimise_pairs_trading_tickers(
            top_companies,
            start_date,
            end_date,
            strategy_params,
            optimise,
            progress_callback=progress_callback,
            status_callback=_display_status_message,
            load_two_tickers=_cached_load_yfinance_data_two_tickers,
            same_company_check=_cached_is_same_company,
        )
    ticker1, ticker2 = ticker

    # Load historical data for the selected pair.
    data = _cached_load_yfinance_data_two_tickers(
        ticker1, ticker2, start_date, end_date
    )
    ticker_display = f"{ticker1} vs. {ticker2}"
    _display_cointegration_result(
        get_training_data(data),
        "training split used for pair selection",
    )

    if optimise and walk_forward:
        strategy_params, _ = _run_optimisation_with_ui(
            data,
            "Pairs Trading",
            parameter_ranges,
            start_date,
            end_date,
            [ticker1, ticker2],
            walk_forward=walk_forward,
        )
    elif optimise and not walk_forward:
        st.warning(
            "Parameter optimisation without walk-forward validation "
            "uses a single 70/30 train/test split. Results may still "
            "be partially in-sample. Enable walk-forward validation "
            "for more robust out-of-sample evaluation."
        )
        strategy_params = selected_strategy_params
    else:
        strategy_params = selected_strategy_params

    # Calculate and display the time taken for optimisation.
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    st.header("Exploratory Current-Universe Result")
    tickers_and_strategy_params = {
        "ticker1": ticker1,
        "ticker2": ticker2,
    } | strategy_params
    st.write(tickers_and_strategy_params)

    return data, ticker_display, strategy_params


def prepare_pairs_trading_strategy_without_optimisation(
    ticker: tuple[str, str],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
    walk_forward: bool = False,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the pairs trading strategy for user-selected tickers.

    Loads historical data for the user-selected pair of tickers and optimises
    the strategy parameters if requested.

    Args:
        ticker: A tuple containing two ticker symbols.
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.
        walk_forward: Whether to use walk-forward validation.

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    ticker1, ticker2 = ticker
    data = _cached_load_yfinance_data_two_tickers(
        ticker1, ticker2, start_date, end_date
    )
    ticker_display = f"{ticker1} vs. {ticker2}"
    _display_cointegration_result(data, "selected date range")

    if optimise:
        strategy_params, _ = _run_optimisation_with_ui(
            data,
            "Pairs Trading",
            strategy_params,
            start_date,
            end_date,
            [ticker1, ticker2],
            walk_forward=walk_forward,
        )

    return data, ticker_display, strategy_params


def prepare_single_ticker_strategy(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
    optimise: bool,
    walk_forward: bool = False,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles strategies for a single ticker.

    Loads historical data for a single ticker and optimises the strategy
    parameters if requested.

    Args:
        ticker: The ticker symbol.
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.
        walk_forward: Whether to use walk-forward validation.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The ticker symbol.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    data = _cached_load_yfinance_data_one_ticker(ticker, start_date, end_date)
    ticker_display = ticker

    if optimise and strategy_type != "Buy and Hold":
        strategy_params, _ = _run_optimisation_with_ui(
            data,
            strategy_type,
            strategy_params,
            start_date,
            end_date,
            ticker,
            walk_forward=walk_forward,
        )
    elif optimise and strategy_type == "Buy and Hold":
        top_companies = _cached_get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)
        with _streamlit_progress() as progress_callback:
            best_ticker, strategy_params, _ = optimise_buy_and_hold_ticker(
                top_companies,
                start_date,
                end_date,
                progress_callback=progress_callback,
                status_callback=_display_status_message,
                load_one_ticker=_cached_load_yfinance_data_one_ticker,
            )
        ticker = best_ticker
        ticker_display = best_ticker
        data = _cached_load_yfinance_data_one_ticker(ticker, start_date, end_date)

    return data, ticker_display, strategy_params


def _is_streamlit_cloud() -> bool:
    """Return True when running on Streamlit Cloud or a read-only directory."""
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"):
        return True
    return not os.access(Path.cwd(), os.W_OK)


def _get_strategy_store() -> StrategyStore:
    """Select an in-memory store on Streamlit Cloud or SQLite locally."""
    if _is_streamlit_cloud():
        records = cast(
            list[dict[str, Any]],
            st.session_state.setdefault("strategy_results", []),
        )
        return InMemoryStrategyStore(records)
    return SQLiteStrategyStore()


def _build_save_key(
    strategy_type: str,
    strategy_params: dict[str, Any],
    tickers: str | list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    optimise: bool,
    walk_forward: bool,
    auto_select_tickers: bool,
) -> str:
    """Build a canonical deduplication key for Streamlit session reruns."""
    payload = {
        "strategy_type": strategy_type,
        "strategy_params": strategy_params,
        "tickers": tickers,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "optimise": optimise,
        "walk_forward": walk_forward,
        "auto_select_tickers": auto_select_tickers,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _format_historical_sharpe(sharpe_ratio: float | None) -> str:
    """Format a historical Sharpe ratio for display in the results list."""
    if sharpe_ratio is not None and math.isfinite(sharpe_ratio):
        return f"{sharpe_ratio:.2f}"
    return "N/A"


def display_historical_results(store: StrategyStore | None = None) -> None:
    """Display historical strategy results from the active strategy store.

    Args:
        store: Strategy store to read from. Defaults to the environment store.
    """
    target_store = store if store is not None else _get_strategy_store()
    strategies = target_store.list_recent()

    if not strategies:
        st.info("No historical strategy results available.")
        return

    if isinstance(target_store, InMemoryStrategyStore):
        st.info(
            """
            **Note about Results History:**
            - Strategy results are saved within your current session
            - Results will be available as long as you keep this tab open
            - Results are reset when you refresh the page or start a new session
            """
        )

    st.header("Historical Strategy Results")

    # Display stored strategy records newest first.
    for strategy in strategies:
        ticker_display = (
            " vs. ".join(str(t) for t in strategy.tickers)
            if isinstance(strategy.tickers, list)
            else strategy.tickers
        )
        timestamp_display = strategy.date_created.strftime("%Y-%m-%d %H:%M:%S")

        with st.expander(f"{strategy.name} - {ticker_display} - {timestamp_display}"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Strategy Details")
                st.write(f"**Strategy Type:** {strategy.name}")
                st.write(f"**Ticker(s):** {ticker_display}")
                st.write(f"**Date Created:** {timestamp_display}")
                st.write(f"**Start Date:** {strategy.start_date}")
                st.write(f"**End Date:** {strategy.end_date}")

                st.subheader("Parameters")
                for key, value in (strategy.parameters or {}).items():
                    st.write(f"**{key}:** {value}")

            with col2:
                st.subheader("Performance Metrics")
                st.write(f"**Total Return:** {strategy.total_return:.2%}")
                st.write(
                    f"**Sharpe Ratio:** "
                    f"{_format_historical_sharpe(strategy.sharpe_ratio)}"
                )
                st.write(f"**Max Drawdown:** {strategy.max_drawdown:.2%}")

            st.write("---")


def main():
    """
    Orchestrates the Streamlit app flow.

    Sets up the user interface, collects inputs, runs the backtest, and
    displays the results.
    """
    st.title("Quant Trading Strategy Backtester")

    # Get user inputs
    ticker, start_date, end_date, strategy_type, auto_select_tickers = (
        get_user_inputs_except_strategy_params()
    )
    optimise, walk_forward, strategy_params = get_user_inputs_for_strategy_params(
        strategy_type,
        auto_select_tickers,
    )

    # Initialise company names
    company_name1 = None
    company_name2 = None

    # Prepare the trading strategy based on user inputs
    if strategy_type == "Pairs Trading" and auto_select_tickers:
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_with_optimisation(
                start_date,
                end_date,
                strategy_params,
                optimise,
                walk_forward,
            )
        )
        # Update company names with the selected pair.
        ticker1, ticker2 = ticker_display.split(" vs. ")
        company_name1 = _cached_get_full_company_name(ticker1)
        company_name2 = _cached_get_full_company_name(ticker2)
    elif strategy_type == "Pairs Trading":
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_without_optimisation(
                cast(tuple[str, str], ticker),
                start_date,
                end_date,
                strategy_params,
                optimise,
                walk_forward,
            )
        )
        ticker1, ticker2 = cast(tuple[str, str], ticker)
        company_name1 = _cached_get_full_company_name(ticker1)
        company_name2 = _cached_get_full_company_name(ticker2)
    # Optimise the ticker for single ticker strategies if option is selected.
    elif (
        strategy_type in ["Buy and Hold", "Mean Reversion", "Moving Average Crossover"]
        and auto_select_tickers
    ):
        data, ticker_display, strategy_params = (
            prepare_single_ticker_strategy_with_optimisation(
                start_date,
                end_date,
                strategy_type,
                strategy_params,
                optimise,
                walk_forward,
            )
        )
        company_name1 = _cached_get_full_company_name(ticker_display)
    else:
        data, ticker_display, strategy_params = prepare_single_ticker_strategy(
            cast(str, ticker),
            start_date,
            end_date,
            strategy_type,
            strategy_params,
            optimise,
            walk_forward,
        )
        company_name1 = _cached_get_full_company_name(ticker_display)

    if data is None or data.is_empty():
        st.write("No data available for the selected ticker and date range")
        return

    # Create a display name for the company or companies
    if company_name2:
        company_display = f"{company_name1} vs. {company_name2}"
    else:
        company_display = company_name1

    # Run the backtest and display the results
    tickers = (
        ticker_display.split(" vs. ")
        if strategy_type == "Pairs Trading"
        else ticker_display
    )
    use_validation_data = auto_select_tickers or optimise
    backtest_data = _get_final_backtest_data(
        data, use_validation_data, walk_forward=walk_forward
    )
    if backtest_data.is_empty():
        st.write("No validation data available for the selected ticker and date range")
        return

    if use_validation_data:
        results, metrics = _cached_evaluate_validation_period(
            data,
            backtest_data,
            strategy_type,
            strategy_params,
            tickers,
        )
    else:
        results, metrics = _cached_run_backtest(
            backtest_data,
            strategy_type,
            strategy_params,
            tickers,
        )

    # Save only the final backtest result, and only once per unique
    # combination of inputs (not on every Streamlit rerun).
    strategy_store = _get_strategy_store()
    _save_key = _build_save_key(
        strategy_type,
        strategy_params,
        tickers,
        start_date,
        end_date,
        optimise,
        walk_forward,
        auto_select_tickers,
    )
    if st.session_state.get("_last_saved_key") != _save_key:
        strategy = create_strategy(strategy_type, strategy_params)
        backtester = Backtester(backtest_data, strategy, tickers=tickers)
        backtester.results = results
        backtester.save_results(store=strategy_store)
        st.session_state["_last_saved_key"] = _save_key

    if use_validation_data and metrics.get("Trade Events", 1.0) == 0:
        st.warning(
            "The selected strategy produced no trades over the displayed "
            "validation period, so risk-adjusted metrics may be undefined."
        )

    display_performance_metrics(metrics, company_display)
    _display_benchmark_comparison(results, backtest_data, full_data=data)
    plot_equity_curve(
        results,
        ticker_display,
        company_display,
        is_pairs=strategy_type == "Pairs Trading",
    )
    if strategy_type == "Pairs Trading":
        plot_pairs_spread(
            results,
            ticker_display,
            company_display,
            entry_z_score=float(strategy_params.get("entry_z_score", 2.0)),
            exit_z_score=float(strategy_params.get("exit_z_score", 0.5)),
        )
    plot_strategy_returns(results, ticker_display, company_display)
    display_returns_by_month(results)
    display_trade_ledger(results)

    # Display the raw data from Yahoo Finance for the backtest period
    st.header(f"Raw Data for {company_display}")
    st.dataframe(
        backtest_data.to_pandas(),
        width="stretch",
        hide_index=True,
    )

    # Display historical results
    display_historical_results(store=strategy_store)


if __name__ == "__main__":
    main()
