"""
Contains functions related to optimisation and backtesting for strategy
parameters and ticker pairs.
"""

import datetime
import itertools
import time
from typing import Any, cast

import polars as pl
import streamlit as st
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.data import (
    get_top_sp500_companies,
    is_same_company,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)
from quant_trading_strategy_backtester.strategy_templates import (
    TRADING_STRATEGIES,
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
    Strategy,
)
from quant_trading_strategy_backtester.utils import NUM_TOP_COMPANIES_ONE_TICKER


def run_optimisation(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Runs the optimisation process for strategy parameters or ticker selection.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy being optimised.
        strategy_params: Initial strategy parameters or parameter ranges.
        start_date: Start date for historical data.
        end_date: End date for historical data.

    Returns:
        A tuple containing:
            - Optimised strategy parameters or selected ticker.
            - Performance metrics for the optimised strategy.
    """
    st.info("Optimising strategy. This may take a while...")
    start_time = time.time()

    if strategy_type == "Buy and Hold":
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)
        best_ticker, strategy_params, metrics = optimise_buy_and_hold_ticker(
            top_companies, start_date, end_date
        )
        st.success(f"Best ticker for Buy and Hold: {best_ticker}")
    else:
        strategy_params, metrics = optimise_strategy_params(
            data,
            strategy_type,
            cast(dict[str, range] | dict[str, list[float]], strategy_params),
        )

    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    st.header("Optimal Parameters")
    st.write(strategy_params)

    return strategy_params, metrics


def optimise_buy_and_hold_ticker(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """
    Optimises ticker selection for the Buy and Hold strategy.

    Args:
        top_companies: List of tuples containing ticker symbols and market caps
                       of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.

    Returns:
        A tuple containing the best ticker, strategy parameters, and
        performance metrics.
    """
    best_ticker = None
    best_metrics = None
    best_total_return = float("-inf")

    total_tickers = len(top_companies)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (ticker, _) in enumerate(top_companies):
        status_text.text(f"Evaluating ticker {i + 1} / {total_tickers}: {ticker}")
        progress_bar.progress((i + 1) / total_tickers)

        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        if data is None or data.is_empty():
            continue

        strategy = BuyAndHoldStrategy({})
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.get_performance_metrics()

        if metrics and metrics["Total Return"] > best_total_return:
            best_total_return = metrics["Total Return"]
            best_ticker = ticker
            best_metrics = metrics

    progress_bar.empty()
    status_text.empty()

    if not best_ticker or not best_metrics:
        raise ValueError("Buy and Hold optimisation failed")

    return best_ticker, {}, best_metrics


def optimise_strategy_params(
    data: pl.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range] | dict[str, list[float]],
) -> tuple[dict[str, int] | dict[str, float], dict[str, float]]:
    """
    Optimises strategy parameters by testing all combinations within given
    ranges.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy to optimise.
        parameter_ranges: A dictionary of parameters and their possible values
                          to test.

    Returns:
        A tuple containing the best parameters and their performance metrics.
    """
    best_params = None
    best_metrics = None
    best_sharpe_ratio = float("-inf")

    param_names = list(parameter_ranges.keys())
    param_values = [
        list(value) if isinstance(value, range) else value
        for value in parameter_ranges.values()
    ]

    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)
    # Display progress bar and status text, as this process may take a while.
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, params in enumerate(param_combinations):
        status_text.text(
            f"Evaluating parameter combination {i + 1} / {total_combinations}"
        )
        progress_bar.progress((i + 1) / total_combinations)

        current_params = dict(zip(param_names, params))
        _, metrics = run_backtest(data, strategy_type, current_params)

        if metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = metrics["Sharpe Ratio"]
            best_params = current_params
            best_metrics = metrics

    progress_bar.empty()
    status_text.empty()
    if not best_params or not best_metrics:
        raise ValueError("Parameter optimisation failed")

    return best_params, best_metrics


def optimise_pairs_trading_tickers(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, float]]:
    """
    Optimises ticker pair selection and strategy parameters for pairs trading.

    Args:
        top_companies: List of tuples containing ticker symbols and market caps
                       of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        strategy_params: Strategy parameters or parameter ranges.
        optimise: Whether to optimise the strategy parameters.

    Returns:
        A tuple containing the best ticker pair, best parameters, and best
        metrics.
    """
    best_pair = None
    best_params = None
    best_metrics = None
    best_sharpe_ratio = float("-inf")

    ticker_pairs = list(
        itertools.combinations([company[0] for company in top_companies], 2)
    )
    # Filter out pairs that likely represent the same company
    ticker_pairs = [
        pair for pair in ticker_pairs if not is_same_company(pair[0], pair[1])
    ]
    total_combinations = len(ticker_pairs)
    # Display progress bar and status text, as this process may take a while.
    progress_bar = st.progress(0)
    status_text = st.empty()
    prev_pair_processing_time = 0.0

    for i, (ticker1, ticker2) in enumerate(ticker_pairs):
        start_time = time.time()
        status_text.text(
            f"Evaluating pair {i + 1} / {total_combinations}: {ticker1} vs. {ticker2} "
            f"(prev. pair processing time: {prev_pair_processing_time:.4f} seconds)"
        )
        progress_bar.progress((i + 1) / total_combinations)

        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        if data is None or data.is_empty():
            continue

        if optimise:
            # Convert single values to lists for optimisation
            param_ranges = {
                k: [v] if isinstance(v, (int, float)) else v
                for k, v in strategy_params.items()
            }
            current_params, current_metrics = optimise_strategy_params(
                data, "Pairs Trading", param_ranges
            )
        else:
            _, current_metrics = run_backtest(data, "Pairs Trading", strategy_params)
            current_params = strategy_params

        if current_metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = current_metrics["Sharpe Ratio"]
            best_pair = (ticker1, ticker2)
            best_params = current_params
            best_metrics = current_metrics

        end_time = time.time()
        prev_pair_processing_time = end_time - start_time

    progress_bar.empty()
    status_text.empty()
    if not best_pair or not best_params or not best_metrics:
        raise ValueError("Pairs trading optimisation failed")

    return best_pair, best_params, best_metrics


def run_backtest(
    data: pl.DataFrame, strategy_type: str, strategy_params: dict[str, Any]
) -> tuple[pl.DataFrame, dict]:
    """
    Executes the backtest using the selected strategy and parameters.

    Args:
        data: Historical stock data.
        strategy_type: The type of strategy to use for the backtest.
        strategy_params: Additional parameters required for the strategy.

    Returns:
        A tuple containing the backtest results DataFrame and performance metrics.
    """
    strategy = create_strategy(strategy_type, strategy_params)
    backtester = Backtester(data, strategy)
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert (
        metrics is not None
    ), "No results available for the selected ticker and date range"

    return results, metrics


def create_strategy(strategy_type: str, strategy_params: dict[str, Any]) -> Strategy:
    """
    Creates a trading strategy object based on the selected strategy type.

    Args:
        strategy_type: The type of trading strategy.
        strategy_params: A dictionary containing the strategy parameters.
    """
    if strategy_type not in TRADING_STRATEGIES:
        raise ValueError("Invalid strategy type")

    match strategy_type:
        case "Buy and Hold":
            return BuyAndHoldStrategy(strategy_params)
        case "Moving Average Crossover":
            return MovingAverageCrossoverStrategy(strategy_params)
        case "Mean Reversion":
            return MeanReversionStrategy(strategy_params)
        case "Pairs Trading":
            return PairsTradingStrategy(strategy_params)
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")
