"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""

import datetime
import time
from typing import Any, cast

import polars as pl
import streamlit as st
from quant_trading_strategy_backtester.data import (
    get_full_company_name,
    get_top_sp500_companies,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)
from quant_trading_strategy_backtester.optimiser import (
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    run_backtest,
    run_optimisation,
)
from quant_trading_strategy_backtester.streamlit_ui import (
    get_user_inputs_except_strategy_params,
    get_user_inputs_for_strategy_params,
)
from quant_trading_strategy_backtester.utils import NUM_TOP_COMPANIES
from quant_trading_strategy_backtester.visualisation import (
    display_monthly_performance_table,
    display_performance_metrics,
    plot_equity_curve,
    plot_strategy_returns,
)

# Trading strategy preparation functions


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
    st.info(
        f"Selecting the best ticker from the top {NUM_TOP_COMPANIES} S&P 500 "
        "companies. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES)

    # Optimise ticker selection
    best_ticker, _, _ = optimise_buy_and_hold_ticker(
        top_companies, start_date, end_date
    )

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    # Display the optimal ticker
    st.header("Optimal Ticker")
    st.write(f"Best performing ticker: {best_ticker}")

    # Load historical data for the selected ticker
    data = load_yfinance_data_one_ticker(best_ticker, start_date, end_date)

    return data, best_ticker, {}


def prepare_pairs_trading_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
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

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters.
    """
    # Inform the user that the optimisation process is starting
    st.info(
        f"Selecting the best pair from the top {NUM_TOP_COMPANIES} S&P 500 "
        "companies. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES)

    # Optimise ticker pair selection and strategy parameters
    ticker, strategy_params, _ = optimise_pairs_trading_tickers(
        top_companies, start_date, end_date, strategy_params, optimise
    )
    ticker1, ticker2 = ticker

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    # Display the optimal tickers and parameters
    st.header("Optimal Tickers and Parameters")
    tickers_and_strategy_params = {
        "ticker1": ticker1,
        "ticker2": ticker2,
    } | strategy_params
    st.write(tickers_and_strategy_params)

    # Load historical data for the selected pair
    data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
    ticker_display = f"{ticker1} vs. {ticker2}"

    return data, ticker_display, strategy_params


def prepare_pairs_trading_strategy_without_optimisation(
    ticker: tuple[str, str],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
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

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    ticker1, ticker2 = ticker
    data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
    ticker_display = f"{ticker1} vs. {ticker2}"

    if optimise:
        strategy_params, _ = run_optimisation(
            data, "Pairs Trading", strategy_params, start_date, end_date
        )

    return data, ticker_display, strategy_params


def prepare_single_ticker_strategy(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
    optimise: bool,
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

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The ticker symbol.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
    ticker_display = ticker

    if optimise and strategy_type != "Buy and Hold":
        strategy_params, _ = run_optimisation(
            data, strategy_type, strategy_params, start_date, end_date
        )
    elif optimise and strategy_type == "Buy and Hold":
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES)
        best_ticker, strategy_params, _ = optimise_buy_and_hold_ticker(
            top_companies, start_date, end_date
        )
        ticker = best_ticker
        ticker_display = best_ticker
        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)

    return data, ticker_display, strategy_params


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
    optimise, strategy_params = get_user_inputs_for_strategy_params(strategy_type)

    # Initialise company names
    company_name1 = None
    company_name2 = None

    # Prepare the trading strategy based on user inputs
    if strategy_type == "Pairs Trading" and auto_select_tickers:
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_with_optimisation(
                start_date, end_date, strategy_params, optimise
            )
        )
        # Update company names with the selected pair
        ticker1, ticker2 = ticker_display.split(" vs. ")
        company_name1 = get_full_company_name(ticker1)
        company_name2 = get_full_company_name(ticker2)
    elif strategy_type == "Pairs Trading":
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_without_optimisation(
                cast(tuple[str, str], ticker),
                start_date,
                end_date,
                strategy_params,
                optimise,
            )
        )
        ticker1, ticker2 = cast(tuple[str, str], ticker)
        company_name1 = get_full_company_name(ticker1)
        company_name2 = get_full_company_name(ticker2)
    elif strategy_type == "Buy and Hold" and auto_select_tickers:
        data, ticker_display, strategy_params = (
            prepare_buy_and_hold_strategy_with_optimisation(start_date, end_date)
        )
        company_name1 = get_full_company_name(ticker_display)
    else:
        data, ticker_display, strategy_params = prepare_single_ticker_strategy(
            cast(str, ticker),
            start_date,
            end_date,
            strategy_type,
            strategy_params,
            optimise,
        )
        company_name1 = get_full_company_name(ticker_display)

    if data is None or data.is_empty():
        st.write("No data available for the selected ticker and date range")
        return

    # Create a display name for the company or companies
    if company_name2:
        company_display = f"{company_name1} vs. {company_name2}"
    else:
        company_display = company_name1

    # Run the backtest and display the results
    results, metrics = run_backtest(data, strategy_type, strategy_params)
    display_performance_metrics(metrics, company_display)
    plot_equity_curve(results, ticker_display, company_display)
    plot_strategy_returns(results, ticker_display, company_display)
    display_monthly_performance_table(results)

    # Display the raw data from Yahoo Finance for the backtest period
    st.header(f"Raw Data for {company_display}")
    st.dataframe(
        data.to_pandas(),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
