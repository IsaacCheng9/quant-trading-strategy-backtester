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
