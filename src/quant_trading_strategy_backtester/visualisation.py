"""
Contains functions to display backtest results using Streamlit and Plotly.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st


def display_performance_metrics(
    metrics: dict[str, float], company_name: str | None
) -> None:
    """
    Displays key performance metrics of the backtest.

    Args:
        metrics: A dictionary containing performance metrics.
        company_name: The full name of the company or companies.
    """
    st.header(f"Backtest Results for {company_name}")
    total_return_col, sharpe_ratio_col, max_drawdown_col = st.columns(3)
    total_return_col.metric("Total Return", f"{metrics['Total Return']:.4%}")
    sharpe_ratio_col.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.4f}")
    max_drawdown_col.metric("Max Drawdown", f"{metrics['Max Drawdown']:.4%}")


def plot_equity_curve(
    results: pl.DataFrame, ticker_display: str, company_name: str | None
) -> None:
    """
    Plots the equity curve of the backtest.

    Args:
        results: The backtest results DataFrame.
        ticker_display: The stock ticker symbol or pair to display.
        company_name: The full name of the company or companies.
    """
    st.subheader("Equity Curve")
    fig = go.Figure(
        data=go.Scatter(
            x=results["Date"].to_list(),
            y=results["equity_curve"].to_list(),
            mode="lines",
        )
    )
    fig.update_layout(
        title=f"{company_name} ({ticker_display}) Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )
    st.plotly_chart(fig)


def plot_strategy_returns(
    results: pl.DataFrame, ticker_display: str, company_name: str | None
) -> None:
    """
    Plots the strategy returns over time.

    Args:
        results: The backtest results DataFrame.
        ticker_display: The stock ticker symbol or pair to display.
        company_name: The full name of the company or companies.
    """
    st.subheader("Strategy Returns")
    fig = go.Figure(
        data=go.Scatter(
            x=results["Date"].to_list(),
            y=results["strategy_returns"].to_list(),
            mode="lines",
        )
    )
    fig.update_layout(
        title=f"{company_name} ({ticker_display}) Strategy Returns",
        xaxis_title="Date",
        yaxis_title="Returns",
    )
    st.plotly_chart(fig)
