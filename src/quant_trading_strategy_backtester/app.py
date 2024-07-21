"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""

import datetime
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
)

logger = logging.getLogger(__name__)
NUM_TOP_COMPANIES = 20


@st.cache_data
def load_yfinance_data_one_ticker(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    Fetches historical stock data for a ticker from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol.
        start_date: The start date for the data.
        end_date: The end date for the data.

    Returns:
        A DataFrame containing the historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


@st.cache_data
def load_yfinance_data_two_tickers(
    ticker1: str, ticker2: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    Fetches historical stock data for two tickers from Yahoo Finance.

    Args:
        ticker1: The first stock ticker symbol.
        ticker2: The second stock ticker symbol.
        start_date: The start date for the data.
        end_date: The end date for the data.

    Returns:
        A DataFrame containing the historical stock data for both tickers.
    """
    data1 = yf.download(ticker1, start=start_date, end=end_date)
    data2 = yf.download(ticker2, start=start_date, end=end_date)
    combined_data = pd.DataFrame({"Close_1": data1["Close"], "Close_2": data2["Close"]})
    return combined_data


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
    Fetches the top X companies in the S&P 500 index by market cap using
    yfinance.

    Args:
        num_companies: The number of top companies to fetch.

    Returns:
        A list of tuples containing the ticker symbols and market cap
        of each company in the top X of the S&P 500 index, sorted by market
        cap.
    """
    # Fetch the list of S&P 500 companies from the Wikipedia table.
    SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(SOURCE)[0]
    tickers = sp500["Symbol"].tolist()

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
    top_companies = sorted(sp500_companies, key=lambda x: x[1], reverse=True)[
        :num_companies
    ]

    return top_companies


def get_user_inputs_except_strategy_params() -> (
    tuple[str | tuple[str, str] | None, datetime.date, datetime.date, str, bool]
):
    """
    Gets user inputs besides strategy parameters from the Streamlit sidebar.

    Returns:
        A tuple containing the ticker symbol(s), start date, end date, strategy
        type, and a boolean indicating whether to use automatic ticker
        selection for pairs trading.
    """
    strategy_type = cast(
        str,
        st.sidebar.selectbox(
            "Strategy Type",
            ["Mean Reversion", "Moving Average Crossover", "Pairs Trading"],
            index=0,
        ),
    )

    auto_select_tickers = False
    if strategy_type == "Pairs Trading":
        auto_select_tickers = st.sidebar.checkbox(
            f"Optimise Ticker Pair From Top {NUM_TOP_COMPANIES} S&P 500 Companies"
        )
        if auto_select_tickers:
            ticker = None  # We'll select tickers later
        else:
            ticker1: str = st.sidebar.text_input(
                "Ticker Symbol 1", value="AAPL"
            ).upper()
            ticker2: str = st.sidebar.text_input(
                "Ticker Symbol 2", value="GOOGL"
            ).upper()
            ticker = (ticker1, ticker2)
    else:
        ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

    start_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01")),
    )
    end_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31")),
    )

    return ticker, start_date, end_date, strategy_type, auto_select_tickers


def optimise_pairs_trading_tickers(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, float]]:
    """
    Optimizes ticker pair selection and strategy parameters for pairs trading.

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

    # Generate all possible pairs of tickers
    ticker_pairs = list(
        itertools.combinations([company[0] for company in top_companies], 2)
    )

    for ticker1, ticker2 in ticker_pairs:
        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        if data is None or data.empty:
            continue

        if optimise:
            current_params, current_metrics = optimise_strategy_params(
                data, "Pairs Trading", strategy_params
            )
        else:
            _, current_metrics = run_backtest(data, "Pairs Trading", strategy_params)
            current_params = strategy_params

        if current_metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = current_metrics["Sharpe Ratio"]
            best_pair = (ticker1, ticker2)
            best_params = current_params
            best_metrics = current_metrics

    if best_pair is None:
        raise ValueError("No valid ticker pair found")

    return best_pair, best_params, best_metrics


def get_user_inputs_for_strategy_params(
    strategy_type: str,
) -> tuple[bool, dict[str, float] | dict[str, range] | dict[str, list[float]]]:
    """
    Gets user inputs for the strategy parameters from the Streamlit sidebar
    based on the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.

    Returns:
        A dictionary containing the strategy parameters.
    """
    optimise = st.sidebar.checkbox("Optimise Strategy Parameters")
    if optimise:
        if strategy_type == "Moving Average Crossover":
            params = {
                "short_window": range(5, 51, 5),
                "long_window": range(20, 201, 20),
            }
        elif strategy_type == "Mean Reversion":
            params = {
                "window": range(5, 101, 5),
                "std_dev": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            }
        elif strategy_type == "Pairs Trading":
            params = {
                "window": range(10, 101, 10),
                "entry_z_score": [1.0, 1.5, 2.0, 2.5, 3.0],
                "exit_z_score": [0.1, 0.5, 1.0, 1.5],
            }
    else:
        if strategy_type == "Moving Average Crossover":
            short_window = st.sidebar.slider(
                "Short Window (Days)", min_value=5, max_value=50, value=20
            )
            long_window = st.sidebar.slider(
                "Long Window (Days)", min_value=20, max_value=200, value=50
            )
            params = {"short_window": short_window, "long_window": long_window}
        elif strategy_type == "Mean Reversion":
            window = st.sidebar.slider(
                "Window (Days)", min_value=5, max_value=100, value=20
            )
            std_dev = st.sidebar.slider(
                "Standard Deviation", min_value=0.5, max_value=3.0, value=2.0, step=0.1
            )
            params = {"window": window, "std_dev": std_dev}
        elif strategy_type == "Pairs Trading":
            window = st.sidebar.slider(
                "Window (Days)", min_value=10, max_value=100, value=50
            )
            entry_z_score = st.sidebar.slider(
                "Entry Z-Score", min_value=1.0, max_value=3.0, value=2.0, step=0.1
            )
            exit_z_score = st.sidebar.slider(
                "Exit Z-Score", min_value=0.1, max_value=1.5, value=0.5, step=0.1
            )
            params = {
                "window": window,
                "entry_z_score": entry_z_score,
                "exit_z_score": exit_z_score,
            }

    return optimise, params


def run_backtest(
    data: pd.DataFrame, strategy_type: str, strategy_params: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Executes the backtest using the selected strategy and parameters.

    Args:
        data: Historical stock data.
        strategy_type: The type of strategy to use for the backtest.
        strategy_params: Additional parameters required for the strategy.

    Returns:
        A tuple containing the backtest results DataFrame and performance metrics.
    """
    if strategy_type == "Moving Average Crossover":
        strategy = MovingAverageCrossoverStrategy(strategy_params)
    elif strategy_type == "Mean Reversion":
        strategy = MeanReversionStrategy(strategy_params)
    elif strategy_type == "Pairs Trading":
        strategy = PairsTradingStrategy(strategy_params)
    else:
        raise ValueError("Invalid strategy type")

    backtester = Backtester(data, strategy)
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert (
        metrics is not None
    ), "No results available for the selected ticker and date range."
    return results, metrics


def display_performance_metrics(metrics: dict[str, float]) -> None:
    """
    Displays key performance metrics of the backtest.

    Args:
        metrics: A dictionary containing performance metrics.
    """
    st.header("Backtest Results")
    total_return_col, sharpe_ratio_col, max_drawdown_col = st.columns(3)
    total_return_col.metric("Total Return", f"{metrics['Total Return']:.2%}")
    sharpe_ratio_col.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    max_drawdown_col.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")


def plot_equity_curve(results: pd.DataFrame, ticker_display: str) -> None:
    """
    Plots the equity curve of the backtest.

    Args:
        results: The backtest results DataFrame.
        ticker: The stock ticker symbol.
    """
    st.subheader("Equity Curve")
    fig = go.Figure(
        data=go.Scatter(x=results.index, y=results["equity_curve"], mode="lines")
    )
    fig.update_layout(
        title=f"{ticker_display} Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )
    st.plotly_chart(fig)


def plot_strategy_returns(results: pd.DataFrame, ticker_display: str) -> None:
    """
    Plots the strategy returns over time.

    Args:
        results: The backtest results DataFrame.
        ticker: The stock ticker symbol.
    """
    st.subheader("Strategy Returns")
    fig = go.Figure(
        data=go.Scatter(x=results.index, y=results["strategy_returns"], mode="lines")
    )
    fig.update_layout(
        title=f"{ticker_display} Strategy Returns",
        xaxis_title="Date",
        yaxis_title="Returns",
    )
    st.plotly_chart(fig)


def optimise_strategy_params(
    data: pd.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range] | dict[str, list[float]],
) -> tuple[dict[str, int] | dict[str, float], dict[str, float]]:
    """
    Optimises strategy parameters by testing all combinations within given
    ranges.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy to optimize.
        parameter_ranges: A dictionary of parameters and their possible values
                          to test.

    Returns:
        A tuple containing the best parameters and their performance metrics.
    """
    best_params = None
    best_metrics = None
    best_sharpe_ratio = float("-inf")

    param_names = list(parameter_ranges.keys())
    param_values = []
    for value in parameter_ranges.values():
        if isinstance(value, range):
            param_values.append(list(value))
        elif isinstance(value, list):
            param_values.append(value)
        else:
            raise ValueError(f"Unsupported parameter type: {type(value)}")

    param_combinations = list(itertools.product(*param_values))

    for params in param_combinations:
        current_params = dict(zip(param_names, params))
        _, metrics = run_backtest(data, strategy_type, current_params)

        if metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = metrics["Sharpe Ratio"]
            best_params = current_params
            best_metrics = metrics

    if not best_params or not best_metrics:
        raise ValueError("Parameter optimisation failed")
    return best_params, best_metrics


def main():
    """
    Orchestrates the Streamlit app flow.

    Sets up the user interface, collect inputs, runs the backtest, and displays
    the results.
    """
    st.title("Quant Trading Strategy Backtester")

    # Get user inputs for the backtest and strategy parameters.
    ticker, start_date, end_date, strategy_type, auto_select_tickers = (
        get_user_inputs_except_strategy_params()
    )
    optimise, strategy_params = get_user_inputs_for_strategy_params(strategy_type)

    # Load the historical data from Yahoo Finance.
    if strategy_type == "Pairs Trading" and auto_select_tickers:
        st.info(
            f"Selecting the best pair from the top {NUM_TOP_COMPANIES} S&P "
            "500 companies. This may take a while..."
        )
        start_time = time.time()
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES)
        ticker, strategy_params, metrics = optimise_pairs_trading_tickers(
            top_companies, start_date, end_date, strategy_params, optimise
        )
        ticker1, ticker2 = ticker
        end_time = time.time()
        duration = end_time - start_time
        st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")
        st.header("Optimal Tickers and Parameters")
        tickers_and_strategy_params = {
            "ticker1": ticker1,
            "ticker2": ticker2,
        } | strategy_params
        st.write(tickers_and_strategy_params)
        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        ticker_display = f"{ticker1} vs. {ticker2}"
        results, _ = run_backtest(data, strategy_type, strategy_params)
    elif strategy_type == "Pairs Trading":
        ticker1, ticker2 = ticker
        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        ticker_display = f"{ticker1} vs. {ticker2}"
        if optimise:
            st.info("Optimising parameters. This may take a while...")
            start_time = time.time()
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
        results, metrics = run_backtest(data, strategy_type, strategy_params)
    else:
        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        ticker_display = ticker
        if optimise:
            st.info("Optimising parameters. This may take a while...")
            start_time = time.time()
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
        results, metrics = run_backtest(data, strategy_type, strategy_params)

    if data is None or data.empty:
        st.write("No data available for the selected ticker and date range.")
        return

    # Display results and metrics from the backtest.
    display_performance_metrics(metrics)
    plot_equity_curve(results, ticker_display)
    plot_strategy_returns(results, ticker_display)

    st.header("Raw Data")
    st.write(data)

    if optimise or (strategy_type == "Pairs Trading" and auto_select_tickers):
        st.header("Optimal Parameters")
        st.write(strategy_params)


if __name__ == "__main__":
    main()
