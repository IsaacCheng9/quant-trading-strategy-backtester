"""
Contains functions to display backtest results using Streamlit and Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def display_performance_metrics(metrics: dict[str, float]) -> None:
    """
    Displays key performance metrics of the backtest.

    Args:
        metrics: A dictionary containing performance metrics.
    """
    st.header("Backtest Results")
    total_return_col, sharpe_ratio_col, max_drawdown_col = st.columns(3)
    total_return_col.metric("Total Return", f"{metrics['Total Return']:.4%}")
    sharpe_ratio_col.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.4f}")
    max_drawdown_col.metric("Max Drawdown", f"{metrics['Max Drawdown']:.4%}")


def plot_equity_curve(results: pd.DataFrame, ticker_display: str) -> None:
    """
    Plots the equity curve of the backtest.

    Args:
        results: The backtest results DataFrame.
        ticker_display: The stock ticker symbol or pair to display.
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
        ticker_display: The stock ticker symbol or pair to display.
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
