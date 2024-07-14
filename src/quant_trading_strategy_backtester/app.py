"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""
from typing import cast
import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MovingAverageCrossoverStrategy,
)


@st.cache_data
def load_yfinance_data(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def get_user_inputs() -> tuple[str, datetime.date, datetime.date, int, int]:
    ticker: str = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    start_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01")),
    )
    end_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31")),
    )
    short_window: int = st.sidebar.slider(
        "Short Window", min_value=5, max_value=50, value=20
    )
    long_window: int = st.sidebar.slider(
        "Long Window", min_value=20, max_value=200, value=50
    )
    return ticker, start_date, end_date, short_window, long_window


def run_backtest(
    data: pd.DataFrame, short_window: int, long_window: int
) -> tuple[pd.DataFrame, dict]:
    strategy = MovingAverageCrossoverStrategy(short_window, long_window)
    backtester = Backtester(data, strategy)
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert (
        metrics is not None
    ), "No results available for the selected ticker and date range."
    return results, metrics


def display_metrics(metrics: dict) -> None:
    st.header("Backtest Results")
    total_return_col, sharpe_ratio_col, max_drawdown_col = st.columns(3)
    total_return_col.metric("Total Return", f"{metrics['Total Return']:.2%}")
    sharpe_ratio_col.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    max_drawdown_col.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")


def plot_equity_curve(results: pd.DataFrame, ticker: str) -> None:
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
    st.subheader("Strategy Returns")
    fig = go.Figure(
        data=go.Scatter(x=results.index, y=results["strategy_returns"], mode="lines")
    )
    fig.update_layout(
        title=f"{ticker} Strategy Returns", xaxis_title="Date", yaxis_title="Returns"
    )
    st.plotly_chart(fig)


def main():
    st.title("Quant Trading Strategy Backtester")

    ticker, start_date, end_date, short_window, long_window = get_user_inputs()

    data = load_yfinance_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.write("No data available for the selected ticker and date range.")
        return

    results, metrics = run_backtest(data, short_window, long_window)

    display_metrics(metrics)
    plot_equity_curve(results, ticker)
    plot_strategy_returns(results, ticker)

    st.subheader("Raw Data")
    st.write(data)


if __name__ == "__main__":
    main()
