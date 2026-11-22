"""Collect user inputs from the Streamlit sidebar user interface."""

import datetime
from dataclasses import dataclass
from typing import Any

import streamlit as st

from quant_trading_strategy_backtester.strategies.base import TRADING_STRATEGIES
from quant_trading_strategy_backtester.utils import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSACTION_COST_BPS,
    NUM_TOP_COMPANIES_ONE_TICKER,
    NUM_TOP_COMPANIES_TWO_TICKERS,
)

WALK_FORWARD_HELP_DISABLED = (
    "Unavailable with automatic ticker selection because its selection window "
    "overlaps early walk-forward test folds."
)
WALK_FORWARD_HELP_ENABLED = (
    "Split data into multiple folds and optimise on each training window, "
    "then evaluate out-of-sample. Shows parameter stability over time. "
    "Without this, optimisation uses a single 70/30 train/test split."
)


@dataclass(frozen=True)
class BacktestConfig:
    """User-selected backtest configuration collected from the sidebar."""

    ticker: str | tuple[str, str] | None
    start_date: datetime.date
    end_date: datetime.date
    strategy_type: str
    auto_select_tickers: bool
    optimise: bool
    walk_forward: bool
    strategy_params: dict[str, Any]
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS


def _render_auto_select_checkbox(strategy_type: str) -> bool:
    """Render the automatic ticker selection checkbox in the sidebar."""
    if strategy_type == "Pairs Trading":
        return st.sidebar.checkbox(
            f"Explore Current Top {NUM_TOP_COMPANIES_TWO_TICKERS} S&P 500 Pairs"
        )
    if strategy_type in [
        "Buy and Hold",
        "Mean Reversion",
        "Moving Average Crossover",
    ]:
        return st.sidebar.checkbox(
            f"Explore Current Top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 Companies"
        )

    return False


def _render_ticker_inputs(
    strategy_type: str,
    auto_select_tickers: bool,
    container: Any = None,
) -> str | tuple[str, str] | None:
    """Render ticker symbol input widgets when manual selection is active."""
    target = container if container is not None else st.sidebar
    if auto_select_tickers:
        return None

    if strategy_type == "Pairs Trading":
        ticker1: str = target.text_input("Ticker Symbol 1", value="AAPL").upper()
        ticker2: str = target.text_input("Ticker Symbol 2", value="GOOGL").upper()
        return (ticker1, ticker2)

    return str(target.text_input("Ticker Symbol", value="AAPL").upper())


def get_trading_assumptions(container: Any = None) -> tuple[float, float, float]:
    """Collect initial capital, transaction cost, and slippage inputs.

    Args:
        container: Streamlit container to render widgets into. Defaults to
            st.sidebar.

    Returns:
        A tuple of (initial_capital, transaction_cost_bps, slippage_bps).
    """
    target = container if container is not None else st.sidebar
    target.subheader("Trading Assumptions")
    initial_capital = float(
        target.number_input(
            "Initial Capital ($)",
            min_value=1.0,
            value=DEFAULT_INITIAL_CAPITAL,
            step=1000.0,
            format="%.2f",
        )
    )
    transaction_cost_bps = float(
        target.number_input(
            "Transaction Cost (bps)",
            min_value=0.0,
            value=DEFAULT_TRANSACTION_COST_BPS,
            step=0.5,
            format="%.2f",
        )
    )
    slippage_bps = float(
        target.number_input(
            "Slippage (bps)",
            min_value=0.0,
            value=DEFAULT_SLIPPAGE_BPS,
            step=0.5,
            format="%.2f",
        )
    )
    return initial_capital, transaction_cost_bps, slippage_bps


def get_sidebar_inputs() -> tuple[BacktestConfig, bool]:
    """Render sidebar controls and return the backtest config and submit state.

    Keep dynamic layout controls (Strategy Type, automatic ticker selection,
    and parameter optimisation toggle) outside st.form so toggling them
    immediately reveals the matching fields. Wrap tickers, dates, strategy
    parameters, walk-forward validation, and trading assumptions inside
    st.form with an explicit 'Run Backtest' submit button.

    Returns:
        A tuple of (config, submitted), where submitted is True when the user
        clicked 'Run Backtest' on the current rerun.
    """
    strategy_type = st.sidebar.selectbox("Strategy Type", TRADING_STRATEGIES, index=0)
    auto_select_tickers = _render_auto_select_checkbox(strategy_type)
    optimise = (
        False
        if strategy_type == "Buy and Hold"
        else st.sidebar.checkbox("Optimise Strategy Parameters")
    )

    with st.sidebar.form("backtest_config"):
        ticker = _render_ticker_inputs(
            strategy_type,
            auto_select_tickers,
            container=st,
        )
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
        end_date = st.date_input("End Date", value=datetime.date(2023, 12, 31))

        walk_forward = False
        strategy_params: dict[str, Any]
        if strategy_type == "Buy and Hold":
            strategy_params = {}
        elif optimise:
            strategy_params = get_optimisation_ranges(strategy_type)
            walk_forward_checked = st.checkbox(
                "Use Walk-Forward Validation",
                help=(
                    WALK_FORWARD_HELP_DISABLED
                    if auto_select_tickers
                    else WALK_FORWARD_HELP_ENABLED
                ),
                disabled=auto_select_tickers,
            )
            walk_forward = False if auto_select_tickers else walk_forward_checked
        else:
            strategy_params = get_fixed_params(strategy_type, container=st)

        initial_capital, transaction_cost_bps, slippage_bps = get_trading_assumptions(
            container=st
        )
        submitted = st.form_submit_button("Run Backtest")

    config = BacktestConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        strategy_type=strategy_type,
        auto_select_tickers=auto_select_tickers,
        optimise=optimise,
        walk_forward=walk_forward,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    return config, submitted


def get_optimisation_ranges(strategy_type: str) -> dict[str, Any]:
    """Return parameter ranges for optimisation based on the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.

    Returns:
        A dictionary containing the parameter ranges for optimisation.
    """
    if strategy_type not in TRADING_STRATEGIES:
        raise ValueError("Invalid strategy type")

    match strategy_type:
        case "Moving Average Crossover":
            return {
                "short_window": range(5, 51, 5),
                "long_window": range(20, 201, 20),
            }
        case "Mean Reversion":
            return {
                "window": range(5, 101, 5),
                "std_dev": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            }
        case "Pairs Trading":
            return {
                "window": range(10, 101, 10),
                "entry_z_score": [1.0, 1.5, 2.0, 2.5, 3.0],
                "exit_z_score": [0.1, 0.5, 1.0, 1.5],
            }
        case "Buy and Hold":
            return {}
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")


def get_fixed_params(strategy_type: str, container: Any = None) -> dict[str, Any]:
    """Collect fixed, user-defined parameters for the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.
        container: Streamlit container to render sliders into. Defaults to
            st.sidebar.

    Returns:
        A dictionary containing the fixed parameters for the strategy.
    """
    if strategy_type not in TRADING_STRATEGIES:
        raise ValueError("Invalid strategy type")

    target = container if container is not None else st.sidebar
    match strategy_type:
        case "Moving Average Crossover":
            short_window = target.slider(
                "Short Window (Days)", min_value=5, max_value=50, value=20
            )
            # Keep bounds static so Streamlit does not reset the slider inside
            # st.form when short_window changes; app.py validates on submit.
            long_window = target.slider(
                "Long Window (Days)",
                min_value=20,
                max_value=200,
                value=50,
            )
            return {"short_window": short_window, "long_window": long_window}
        case "Mean Reversion":
            window = target.slider(
                "Window (Days)", min_value=5, max_value=100, value=20
            )
            std_dev = target.slider(
                "Standard Deviation", min_value=0.5, max_value=3.0, value=2.0, step=0.1
            )
            return {"window": window, "std_dev": std_dev}
        case "Pairs Trading":
            window = target.slider(
                "Window (Days)", min_value=10, max_value=100, value=50
            )
            entry_z_score = target.slider(
                "Entry Z-Score", min_value=1.0, max_value=3.0, value=2.0, step=0.1
            )
            # Keep bounds static so Streamlit does not reset the slider inside
            # st.form when entry_z_score changes; app.py validates on submit.
            exit_z_score = target.slider(
                "Exit Z-Score",
                min_value=0.1,
                max_value=1.5,
                value=0.5,
                step=0.1,
            )
            return {
                "window": window,
                "entry_z_score": entry_z_score,
                "exit_z_score": exit_z_score,
            }
        case "Buy and Hold":
            return {}
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")
