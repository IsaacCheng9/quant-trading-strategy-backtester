"""
Contains functions to display backtest results using Streamlit and Plotly.
"""

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from quant_trading_strategy_backtester.backtester import build_trade_ledger


def _format_metric(value: float, format_spec: str) -> str:
    """Format finite metrics and display undefined metrics clearly."""
    if not math.isfinite(value):
        return "N/A"

    return f"{value:{format_spec}}"


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
    total_return_col.metric(
        "Total Return", _format_metric(metrics["Total Return"], ".4%")
    )
    sharpe_ratio_col.metric(
        "Sharpe Ratio", _format_metric(metrics["Sharpe Ratio"], ".4f")
    )
    max_drawdown_col.metric(
        "Max Drawdown", _format_metric(metrics["Max Drawdown"], ".4%")
    )

    if {
        "Sortino Ratio",
        "Calmar Ratio",
        "Max Drawdown Duration",
    }.issubset(metrics):
        sortino_col, calmar_col, drawdown_duration_col = st.columns(3)
        sortino_col.metric(
            "Sortino Ratio", _format_metric(metrics["Sortino Ratio"], ".4f")
        )
        calmar_col.metric(
            "Calmar Ratio", _format_metric(metrics["Calmar Ratio"], ".4f")
        )
        drawdown_duration_col.metric(
            "Max Drawdown Duration",
            _format_metric(metrics["Max Drawdown Duration"], ".0f"),
        )

    if {
        "Gross Total Return",
        "Total Costs",
        "Cost Drag",
        "Trade Events",
        "Total Turnover",
    }.issubset(metrics):
        gross_return_col, cost_drag_col, total_costs_col = st.columns(3)
        gross_return_col.metric(
            "Gross Total Return",
            _format_metric(metrics["Gross Total Return"], ".4%"),
        )
        cost_drag_col.metric("Cost Drag", _format_metric(metrics["Cost Drag"], ".4%"))
        total_costs_col.metric(
            "Total Costs", _format_metric(metrics["Total Costs"], ".4%")
        )

        trade_events_col, turnover_col, _ = st.columns(3)
        trade_events_col.metric(
            "Trade Events", _format_metric(metrics["Trade Events"], ".0f")
        )
        turnover_col.metric(
            "Total Turnover", _format_metric(metrics["Total Turnover"], ".4f")
        )


def display_benchmark_metrics(metrics: dict[str, float], benchmark_ticker: str) -> None:
    """
    Display benchmark-relative performance metrics.

    Args:
        metrics: A dictionary containing benchmark-relative metrics.
        benchmark_ticker: The ticker symbol used as the benchmark.
    """
    st.subheader(f"Benchmark Comparison vs. {benchmark_ticker}")
    benchmark_return_col, excess_return_col, beta_col = st.columns(3)
    benchmark_return_col.metric(
        "Benchmark Return", _format_metric(metrics["Benchmark Total Return"], ".4%")
    )
    excess_return_col.metric(
        "Excess Return", _format_metric(metrics["Excess Return"], ".4%")
    )
    beta_col.metric("Beta", _format_metric(metrics["Beta"], ".4f"))

    alpha_col, information_ratio_col, observations_col = st.columns(3)
    alpha_col.metric("Annualised Alpha", _format_metric(metrics["Alpha"], ".4%"))
    information_ratio_col.metric(
        "Information Ratio", _format_metric(metrics["Information Ratio"], ".4f")
    )
    observations_col.metric(
        "Aligned Days", _format_metric(metrics["Benchmark Observations"], ".0f")
    )


def plot_equity_curve(
    results: pl.DataFrame,
    ticker_display: str,
    company_name: str | None,
    is_pairs: bool = False,
) -> None:
    """
    Plots the equity curve with trade markers overlaid.

    Args:
        results: The backtest results DataFrame.
        ticker_display: The stock ticker symbol or pair to display.
        company_name: The full name of the company or companies.
        is_pairs: Whether this is a pairs trading strategy. Changes
            marker labels from Buy/Sell to Long/Short Spread.
    """
    st.subheader("Equity Curve")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["Date"].to_list(),
            y=results["equity_curve"].to_list(),
            mode="lines",
            name="Equity",
        )
    )

    # Overlay trade markers where position changes.
    long_label = "Long Spread" if is_pairs else "Buy"
    short_label = "Short Spread" if is_pairs else "Sell"
    buys = results.filter(pl.col("position_change") > 0)
    sells = results.filter(pl.col("position_change") < 0)

    if not buys.is_empty():
        fig.add_trace(
            go.Scatter(
                x=buys["Date"].to_list(),
                y=buys["equity_curve"].to_list(),
                mode="markers",
                marker=dict(size=6, color="green", opacity=0.7),
                name=long_label,
            )
        )
    if not sells.is_empty():
        fig.add_trace(
            go.Scatter(
                x=sells["Date"].to_list(),
                y=sells["equity_curve"].to_list(),
                mode="markers",
                marker=dict(size=6, color="red", opacity=0.7),
                name=short_label,
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


def plot_pairs_spread(
    results: pl.DataFrame,
    ticker_display: str,
    company_name: str | None,
    entry_z_score: float,
    exit_z_score: float,
) -> None:
    """
    Plot the z-score of the pairs spread with entry/exit threshold
    bands and position markers.

    Args:
        results: The backtest results DataFrame (must contain
            'z_score' and 'signal' columns).
        ticker_display: The stock ticker pair to display.
        company_name: The full name of the companies.
        entry_z_score: The z-score threshold for entering a trade.
        exit_z_score: The z-score threshold for exiting a trade.
    """
    if "z_score" not in results.columns:
        return

    st.subheader("Pairs Spread Z-Score")
    dates = results["Date"].to_list()
    z_scores = results["z_score"].to_list()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=z_scores,
            mode="lines",
            name="Z-Score",
        )
    )

    # Entry/exit threshold bands.
    for val, label, dash in [
        (entry_z_score, f"Entry ({entry_z_score})", "dash"),
        (-entry_z_score, f"Entry ({-entry_z_score})", "dash"),
        (exit_z_score, f"Exit ({exit_z_score})", "dot"),
        (-exit_z_score, f"Exit ({-exit_z_score})", "dot"),
    ]:
        fig.add_hline(
            y=val,
            line_dash=dash,
            line_color="grey",
            opacity=0.6,
            annotation_text=label,
        )

    # Mark position changes on the z-score line.
    buys = results.filter(pl.col("position_change") > 0)
    sells = results.filter(pl.col("position_change") < 0)

    if not buys.is_empty():
        fig.add_trace(
            go.Scatter(
                x=buys["Date"].to_list(),
                y=buys["z_score"].to_list(),
                mode="markers",
                marker=dict(size=6, color="green", opacity=0.7),
                name="Long Spread",
            )
        )
    if not sells.is_empty():
        fig.add_trace(
            go.Scatter(
                x=sells["Date"].to_list(),
                y=sells["z_score"].to_list(),
                mode="markers",
                marker=dict(size=6, color="red", opacity=0.7),
                name="Short Spread",
            )
        )

    fig.update_layout(
        title=(f"{company_name} ({ticker_display}) Pairs Spread Z-Score"),
        xaxis_title="Date",
        yaxis_title="Z-Score",
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

    monthly_returns: pl.DataFrame = (  # type: ignore[invalid-assignment]
        results.lazy()
        .with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("Month (YYYY-MM)"))
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
        .collect()
    )

    if monthly_returns.is_empty():
        st.write("No monthly data available after aggregation.")
    else:
        initial_start_value = monthly_returns["start_value"][0]
        monthly_returns: pl.DataFrame = (  # type: ignore[invalid-assignment]
            monthly_returns.lazy()
            .with_columns(
                ((pl.col("end_value") / initial_start_value - 1) * 100).alias(
                    "Rolling Return (%)"
                )
            )
            .with_columns(
                [
                    pl.col("Monthly Return (%)").round(2),
                    pl.col("Rolling Return (%)").round(2),
                ]
            )
            .collect()
        )
        # Display the table
        st.dataframe(
            monthly_returns.select(
                ["Month (YYYY-MM)", "Monthly Return (%)", "Rolling Return (%)"]
            ).to_pandas(),
            width="content",
            hide_index=True,
        )


def display_trade_ledger(results: pl.DataFrame) -> None:
    """
    Display cost attribution and trade ledger rows from backtest results.

    Args:
        results: The backtest results DataFrame.
    """
    st.subheader("Cost Attribution")
    if results.is_empty():
        st.write("No data available for cost attribution.")
        return

    total_costs = float(results["transaction_costs"].cast(pl.Float64).sum())
    total_turnover = float(results["trade_turnover"].cast(pl.Float64).sum())
    trade_events = results.filter(pl.col("trade_turnover") > 0).height

    costs_col, turnover_col, events_col = st.columns(3)
    costs_col.metric("Total Costs", f"{total_costs:.4%}")
    turnover_col.metric("Total Turnover", f"{total_turnover:.4f}")
    events_col.metric("Trade Events", f"{trade_events}")

    st.subheader("Trade Ledger")
    ledger = build_trade_ledger(results)
    if ledger.is_empty():
        st.write("No trades were generated for this backtest.")
        return

    display_columns = [
        "Date",
        "Action",
        "Reason",
        "Signal",
        "Turnover",
        "Gross Return",
        "Transaction Costs",
        "Net Return",
        "Cumulative Costs",
        "Equity",
        "Holding Period Days",
    ]
    optional_columns = ["Leg 1 Weight", "Leg 2 Weight", "Z-Score"]
    display_columns.extend(
        column
        for column in optional_columns
        if column in ledger.columns and ledger[column].null_count() < ledger.height
    )

    ledger_display = ledger.select(display_columns).with_columns(
        [
            pl.col("Turnover").round(4),
            (pl.col("Gross Return") * 100).round(4).alias("Gross Return (%)"),
            (pl.col("Transaction Costs") * 100).round(4).alias("Transaction Costs (%)"),
            (pl.col("Net Return") * 100).round(4).alias("Net Return (%)"),
            (pl.col("Cumulative Costs") * 100).round(4).alias("Cumulative Costs (%)"),
            pl.col("Equity").round(2),
        ]
    )
    ledger_display = ledger_display.drop(
        [
            "Gross Return",
            "Transaction Costs",
            "Net Return",
            "Cumulative Costs",
        ]
    )

    st.dataframe(
        ledger_display.to_pandas(),
        width="stretch",
        hide_index=True,
    )
