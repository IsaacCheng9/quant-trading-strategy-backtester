"""
Contains functions to get user inputs from the Streamlit sidebar user
interface.
"""

import datetime
from typing import Any, cast

import streamlit as st
from quant_trading_strategy_backtester.strategy_templates import TRADING_STRATEGIES
from quant_trading_strategy_backtester.utils import (
    NUM_TOP_COMPANIES_ONE_TICKER,
    NUM_TOP_COMPANIES_TWO_TICKERS,
)


def get_user_inputs_except_strategy_params() -> (
    tuple[str | tuple[str, str] | None, datetime.date, datetime.date, str, bool]
):
    """
    Gets user inputs besides strategy parameters from the Streamlit sidebar.

    Returns:
        A tuple containing the ticker symbol(s), start date, end date, strategy
        type, and a boolean indicating whether to use automatic ticker
        selection for pairs trading or buy and hold.
    """
    strategy_type = cast(
        str, st.sidebar.selectbox("Strategy Type", TRADING_STRATEGIES, index=0)
    )

    auto_select_tickers = False
    # Two ticker strategies
    if strategy_type == "Pairs Trading":
        auto_select_tickers = st.sidebar.checkbox(
            f"Optimise Ticker Pair From Top {NUM_TOP_COMPANIES_TWO_TICKERS} S&P 500 "
            "Companies"
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
    # One ticker strategies
    elif strategy_type in [
        "Buy and Hold",
        "Mean Reversion",
        "Moving Average Crossover",
    ]:
        auto_select_tickers = st.sidebar.checkbox(
            f"Optimise Ticker From Top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 Companies"
        )
        if auto_select_tickers:
            ticker = None  # We'll select the ticker later
        else:
            ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    else:
        ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

    start_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1)),
    )
    end_date: datetime.date = cast(
        datetime.date,
        st.sidebar.date_input("End Date", value=datetime.date(2023, 12, 31)),
    )

    return ticker, start_date, end_date, strategy_type, auto_select_tickers


def get_optimisation_ranges(strategy_type: str) -> dict[str, Any]:
    """
    Gets the parameter ranges for optimisation based on the selected strategy.

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
            return {}  # No parameters to optimise
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")


def get_fixed_params(strategy_type: str) -> dict[str, Any]:
    """
    Gets the fixed, user-defined parameters for the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.

    Returns:
        A dictionary containing the fixed parameters for the strategy.
    """
    if strategy_type not in TRADING_STRATEGIES:
        raise ValueError("Invalid strategy type")

    match strategy_type:
        case "Moving Average Crossover":
            short_window = st.sidebar.slider(
                "Short Window (Days)", min_value=5, max_value=50, value=20
            )
            long_window = st.sidebar.slider(
                "Long Window (Days)", min_value=20, max_value=200, value=50
            )
            return {"short_window": short_window, "long_window": long_window}
        case "Mean Reversion":
            window = st.sidebar.slider(
                "Window (Days)", min_value=5, max_value=100, value=20
            )
            std_dev = st.sidebar.slider(
                "Standard Deviation", min_value=0.5, max_value=3.0, value=2.0, step=0.1
            )
            return {"window": window, "std_dev": std_dev}
        case "Pairs Trading":
            window = st.sidebar.slider(
                "Window (Days)", min_value=10, max_value=100, value=50
            )
            entry_z_score = st.sidebar.slider(
                "Entry Z-Score", min_value=1.0, max_value=3.0, value=2.0, step=0.1
            )
            exit_z_score = st.sidebar.slider(
                "Exit Z-Score", min_value=0.1, max_value=1.5, value=0.5, step=0.1
            )
            return {
                "window": window,
                "entry_z_score": entry_z_score,
                "exit_z_score": exit_z_score,
            }
        case "Buy and Hold":
            return {}  # No parameters needed
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")


def get_user_inputs_for_strategy_params(
    strategy_type: str,
) -> tuple[bool, dict[str, float] | dict[str, range] | dict[str, list[float]]]:
    """
    Gets user inputs for the strategy parameters from the Streamlit sidebar
    based on the selected strategy.

    Args:
        strategy_type: The type of strategy selected by the user.

    Returns:
        A tuple containing a boolean indicating whether to optimise, and a dictionary of strategy parameters.
    """
    if strategy_type == "Buy and Hold":
        return False, {}  # No parameters for Buy and Hold strategy

    optimise = st.sidebar.checkbox("Optimise Strategy Parameters")

    if optimise:
        params = get_optimisation_ranges(strategy_type)
    else:
        params = get_fixed_params(strategy_type)

    return optimise, params
