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
            y=(results["strategy_returns"] * 100).to_list(),  # Convert to percentage
            mode="lines",
        )
    )
    fig.update_layout(
        title=f"{company_name} ({ticker_display}) Strategy Daily Returns",
        xaxis_title="Date",
        yaxis_title="Returns (%)",
        yaxis_tickformat=".2f",  # Format y-axis ticks to 2 decimal places
    )
    st.plotly_chart(fig)


def display_returns_by_month(results: pl.DataFrame) -> None:
    """
    Displays a table showing returns data for each month in the backtest,
    including monthly returns and rolling (cumulative) returns.

    Args:
        results: The backtest results DataFrame.
    """
    st.subheader("Returns by Month")
    if results.is_empty():
        st.write("No data available for monthly performance calculation.")
        return

    monthly_returns = (
        results.with_columns(
            pl.col("Date").dt.strftime("%Y-%m").alias("Month (YYYY-MM)")
        )
        .group_by("Month (YYYY-MM)")
        .agg(
            [
                pl.col("equity_curve").first().alias("start_value"),
                pl.col("equity_curve").last().alias("end_value"),
            ]
        )
        .with_columns(
            (
                (pl.col("end_value") - pl.col("start_value"))
                / pl.col("start_value")
                * 100
            ).alias("Monthly Return (%)")
        )
        .sort("Month (YYYY-MM)")
    )

    if monthly_returns.is_empty():
        st.write("No monthly data available after aggregation.")
    else:
        initial_start_value = monthly_returns["start_value"][0]
        # Calculate rolling returns
        monthly_returns = monthly_returns.with_columns(
            ((pl.col("end_value") / initial_start_value - 1) * 100).alias(
                "Rolling Return (%)"
            )
        )
        # Round the percentage columns
        monthly_returns = monthly_returns.with_columns(
            [
                pl.col("Monthly Return (%)").round(2),
                pl.col("Rolling Return (%)").round(2),
            ]
        )
        # Display the table
        st.dataframe(
            monthly_returns.select(
                ["Month (YYYY-MM)", "Monthly Return (%)", "Rolling Return (%)"]
            ).to_pandas(),
            width="content",
            hide_index=True,
        )
