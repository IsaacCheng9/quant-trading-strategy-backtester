"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""

from typing import Any, cast
import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
)


@st.cache_data
def load_yfinance_data(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol.
        start_date: The start date for the data.
        end_date: The end date for the data.

    Returns:
        A DataFrame containing the historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def get_user_inputs_except_strategy_params() -> (
    tuple[str, datetime.date, datetime.date, str]
):
    """
    Gets user inputs besides strategy parameters from the Streamlit sidebar.

    Returns:
        A tuple containing the ticker symbol, start date, end date, and
        strategy type. The strategy type is used to determine the additional
        input fields that are required.
    """
    ticker: str = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    start_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01")),
    )
    end_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31")),
    )
    # Set the default strategy type to 'Mean Reversion'.
    strategy_type = cast(
        str,
        st.sidebar.selectbox(
            "Strategy Type", ["Mean Reversion", "Moving Average Crossover"], index=0
        ),
    )
    return ticker, start_date, end_date, strategy_type


def get_user_inputs_for_strategy_params(strategy_type: str) -> dict[str, float]:
    """
    Gets user inputs for the strategy parameters from the Streamlit sidebar
    based on the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.

    Returns:
        A dictionary containing the strategy parameters.
    """
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

    return params


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


def plot_equity_curve(results: pd.DataFrame, ticker: str) -> None:
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
        title=f"{ticker} Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )
    st.plotly_chart(fig)


def plot_strategy_returns(results: pd.DataFrame, ticker: str) -> None:
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
        title=f"{ticker} Strategy Returns", xaxis_title="Date", yaxis_title="Returns"
    )
    st.plotly_chart(fig)


def main():
    """
    Orchestrates the Streamlit app flow.

    Sets up the user interface, collect inputs, runs the backtest, and displays
    the results.
    """
    st.title("Quant Trading Strategy Backtester")

    # Get user inputs for the backtest and strategy parameters.
    ticker, start_date, end_date, strategy_type = (
        get_user_inputs_except_strategy_params()
    )
    strategy_params = get_user_inputs_for_strategy_params(strategy_type)

    # Load the historical data and run the backtest.
    data = load_yfinance_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.write("No data available for the selected ticker and date range.")
        return
    results, metrics = run_backtest(data, strategy_type, strategy_params)

    display_performance_metrics(metrics)
    plot_equity_curve(results, ticker)
    plot_strategy_returns(results, ticker)

    st.subheader("Raw Data")
    st.write(data)


if __name__ == "__main__":
    main()
