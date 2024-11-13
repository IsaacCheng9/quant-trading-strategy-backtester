"""
Implements a Streamlit web application for backtesting quantitative trading
strategies.

The application allows users to input a stock ticker, date range, and
parameters for a strategy to backtest it. It displays performance metrics,
equity curve, and strategy returns using interactive Plotly charts.

For instructions on how to run the application, refer to the README.md.
"""

import datetime
import json
import time
from typing import Any, cast

import polars as pl
import streamlit as st

from quant_trading_strategy_backtester.backtester import is_running_locally
from quant_trading_strategy_backtester.data import (
    get_full_company_name,
    get_top_sp500_companies,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)
from quant_trading_strategy_backtester.models import Session, StrategyModel
from quant_trading_strategy_backtester.optimiser import (
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    optimise_single_ticker_strategy_ticker,
    optimise_strategy_params,
    run_backtest,
    run_optimisation,
)
from quant_trading_strategy_backtester.streamlit_ui import (
    get_user_inputs_except_strategy_params,
    get_user_inputs_for_strategy_params,
)
from quant_trading_strategy_backtester.utils import (
    NUM_TOP_COMPANIES_ONE_TICKER,
    NUM_TOP_COMPANIES_TWO_TICKERS,
)
from quant_trading_strategy_backtester.visualisation import (
    display_performance_metrics,
    display_returns_by_month,
    plot_equity_curve,
    plot_strategy_returns,
)

# Trading strategy preparation functions


def prepare_buy_and_hold_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the optimisation process for Buy and Hold strategy.

    Selects the best ticker from the top S&P 500 companies.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The selected ticker symbol.
            - An empty dictionary (no strategy parameters for Buy and Hold).
    """
    st.info(
        f"Selecting the best ticker from the top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 "
        "companies. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)

    # Optimise ticker selection
    best_ticker, _, _ = optimise_buy_and_hold_ticker(
        top_companies, start_date, end_date
    )

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    # Display the optimal ticker
    st.header("Optimal Ticker")
    st.write(f"Best performing ticker: {best_ticker}")

    # Load historical data for the selected ticker
    data = load_yfinance_data_one_ticker(best_ticker, start_date, end_date)

    return data, best_ticker, {}


def prepare_single_ticker_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the optimisation process for single ticker strategies.

    Selects the best ticker from the top S&P 500 companies and optimises
    strategy parameters if requested.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The selected ticker symbol.
            - Optimised strategy parameters.
    """
    st.info(
        f"Selecting the best ticker from the top {NUM_TOP_COMPANIES_ONE_TICKER} S&P 500 "
        "companies. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)

    # Optimise ticker selection
    best_ticker = optimise_single_ticker_strategy_ticker(
        top_companies, start_date, end_date, strategy_type, strategy_params
    )

    # Load historical data for the selected ticker
    data = load_yfinance_data_one_ticker(best_ticker, start_date, end_date)

    # Optimise strategy parameters if requested
    if optimise:
        best_params, _ = optimise_strategy_params(
            data,
            strategy_type,
            cast(dict[str, range | list[int | float]], strategy_params),
            best_ticker,
        )
    else:
        best_params = {
            k: v[0] if isinstance(v, (list, range)) else v
            for k, v in strategy_params.items()
        }

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    # Display the optimal ticker and parameters (if optimised)
    st.header("Optimal Ticker and Parameters")
    if optimise:
        result = {
            "ticker": best_ticker,
            **best_params,
        }
    else:
        result = {
            "ticker": best_ticker,
        }
    st.write(result)

    return data, best_ticker, best_params


def prepare_pairs_trading_strategy_with_optimisation(
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[pl.DataFrame, str, dict[str, int | float]]:
    """
    Handles the optimisation process for pairs trading strategy.

    Selects the best pair of tickers from the top S&P 500 companies and
    optimises the strategy parameters if requested.

    Args:
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters.
    """
    # Inform the user that the optimisation process is starting
    st.info(
        f"Selecting the best pair from the top {NUM_TOP_COMPANIES_TWO_TICKERS} S&P 500 "
        "companies. This may take a while..."
    )

    start_time = time.time()

    # Fetch the top S&P 500 companies
    with st.spinner("Fetching top S&P 500 companies..."):
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_TWO_TICKERS)

    # Optimise ticker pair selection and strategy parameters
    ticker, strategy_params, _ = optimise_pairs_trading_tickers(
        top_companies, start_date, end_date, strategy_params, optimise
    )
    ticker1, ticker2 = ticker

    # Calculate and display the time taken for optimisation
    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    # Display the optimal tickers and parameters
    st.header("Optimal Tickers and Parameters")
    tickers_and_strategy_params = {
        "ticker1": ticker1,
        "ticker2": ticker2,
    } | strategy_params
    st.write(tickers_and_strategy_params)

    # Load historical data for the selected pair
    data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
    ticker_display = f"{ticker1} vs. {ticker2}"

    if optimise:
        strategy_params, _ = run_optimisation(
            data,
            "Pairs Trading",
            strategy_params,
            start_date,
            end_date,
            [ticker1, ticker2],
        )

    return data, ticker_display, strategy_params


def prepare_pairs_trading_strategy_without_optimisation(
    ticker: tuple[str, str],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles the pairs trading strategy for user-selected tickers.

    Loads historical data for the user-selected pair of tickers and optimises
    the strategy parameters if requested.

    Args:
        ticker: A tuple containing two ticker symbols.
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.

    Returns:
        A tuple containing:
            - Historical data for the selected pair.
            - A string representation of the selected pair.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    ticker1, ticker2 = ticker
    data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
    ticker_display = f"{ticker1} vs. {ticker2}"

    if optimise:
        strategy_params, _ = run_optimisation(
            data,
            "Pairs Trading",
            strategy_params,
            start_date,
            end_date,
            [ticker1, ticker2],
        )

    return data, ticker_display, strategy_params


def prepare_single_ticker_strategy(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Handles strategies for a single ticker.

    Loads historical data for a single ticker and optimises the strategy
    parameters if requested.

    Args:
        ticker: The ticker symbol.
        start_date: The start date for historical data.
        end_date: The end date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Initial strategy parameters.
        optimise: Whether to optimise strategy parameters.

    Returns:
        A tuple containing:
            - Historical data for the selected ticker.
            - The ticker symbol.
            - Optimised strategy parameters if optimise is True,
              otherwise the initial strategy parameters.
    """
    data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
    ticker_display = ticker

    if optimise and strategy_type != "Buy and Hold":
        strategy_params, _ = run_optimisation(
            data, strategy_type, strategy_params, start_date, end_date, ticker
        )
    elif optimise and strategy_type == "Buy and Hold":
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)
        best_ticker, strategy_params, _ = optimise_buy_and_hold_ticker(
            top_companies, start_date, end_date
        )
        ticker = best_ticker
        ticker_display = best_ticker
        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)

    return data, ticker_display, strategy_params


def display_historical_results():
    """
    Displays historical strategy results from either the database or session state,
    depending on the environment.
    """
    if is_running_locally():
        # Use SQLite (existing code)
        session = Session()
        strategies = (
            session.query(StrategyModel)
            .order_by(StrategyModel.date_created.desc())
            .all()
        )
        session.close()
    else:
        # Use Streamlit session state
        if "strategy_results" not in st.session_state:
            strategies = []
        else:
            strategies = sorted(
                st.session_state.strategy_results,
                key=lambda x: x["date_created"],
                reverse=True,
            )

    if not strategies:
        st.info("No historical strategy results available.")
        return

    if not is_running_locally():
        st.info("""
            📝 **Note about Results History:**
            - Strategy results are saved within your current session
            - Results will be available as long as you keep this tab open
            - Results are reset when you refresh the page or start a new session
            """)

    st.header("Historical Strategy Results")

    # Display strategies
    for strategy in strategies:
        # Handle either SQLite model or session state dict
        if is_running_locally():
            strategy_name = strategy.name
            date_created = strategy.date_created
            try:
                tickers = json.loads(str(strategy.tickers))
            except (json.JSONDecodeError, TypeError):
                tickers = strategy.tickers
            try:
                params = json.loads(str(strategy.parameters))
            except (json.JSONDecodeError, TypeError):
                params = strategy.parameters  # type: ignore
            total_return = strategy.total_return
            sharpe_ratio: float = strategy.sharpe_ratio  # type: ignore
            max_drawdown = strategy.max_drawdown
            start_date = strategy.start_date
            end_date = strategy.end_date
        else:
            strategy_name = strategy["name"]
            date_created = strategy["date_created"]
            tickers = strategy["tickers"]
            params: dict = strategy["parameters"]
            total_return = strategy["total_return"]
            sharpe_ratio: float = strategy["sharpe_ratio"]
            max_drawdown = strategy["max_drawdown"]
            start_date = strategy["start_date"]
            end_date = strategy["end_date"]

        ticker_display = " vs. ".join(tickers) if isinstance(tickers, list) else tickers

        with st.expander(
            f"{strategy_name} - {ticker_display} - {date_created.strftime('%Y-%m-%d %H:%M:%S')}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Strategy Details")
                st.write(f"**Strategy Type:** {strategy_name}")
                st.write(f"**Ticker(s):** {ticker_display}")
                st.write(
                    f"**Date Created:** {date_created.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                st.write(f"**Start Date:** {start_date}")
                st.write(f"**End Date:** {end_date}")

                st.subheader("Parameters")
                for key, value in (params or {}).items():
                    st.write(f"**{key}:** {value}")

            with col2:
                st.subheader("Performance Metrics")
                st.write(f"**Total Return:** {total_return:.2%}")
                if sharpe_ratio:
                    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                else:
                    st.write("**Sharpe Ratio:** N/A")
                st.write(f"**Max Drawdown:** {max_drawdown:.2%}")

            st.write("---")


def main():
    """
    Orchestrates the Streamlit app flow.

    Sets up the user interface, collects inputs, runs the backtest, and
    displays the results.
    """
    st.title("Quant Trading Strategy Backtester")

    # Get user inputs
    ticker, start_date, end_date, strategy_type, auto_select_tickers = (
        get_user_inputs_except_strategy_params()
    )
    optimise, strategy_params = get_user_inputs_for_strategy_params(strategy_type)

    # Initialise company names
    company_name1 = None
    company_name2 = None

    # Prepare the trading strategy based on user inputs
    if strategy_type == "Pairs Trading" and auto_select_tickers:
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_with_optimisation(
                start_date, end_date, strategy_params, optimise
            )
        )
        # Update company names with the selected pair
        ticker1, ticker2 = ticker_display.split(" vs. ")
        company_name1 = get_full_company_name(ticker1)
        company_name2 = get_full_company_name(ticker2)
    elif strategy_type == "Pairs Trading":
        data, ticker_display, strategy_params = (
            prepare_pairs_trading_strategy_without_optimisation(
                cast(tuple[str, str], ticker),
                start_date,
                end_date,
                strategy_params,
                optimise,
            )
        )
        ticker1, ticker2 = cast(tuple[str, str], ticker)
        company_name1 = get_full_company_name(ticker1)
        company_name2 = get_full_company_name(ticker2)
    # Optimise the ticker for single ticker strategies if option is selected
    elif (
        strategy_type in ["Buy and Hold", "Mean Reversion", "Moving Average Crossover"]
        and auto_select_tickers
    ):
        data, ticker_display, strategy_params = (
            prepare_single_ticker_strategy_with_optimisation(
                start_date, end_date, strategy_type, strategy_params, optimise
            )
        )
        company_name1 = get_full_company_name(ticker_display)
    else:
        data, ticker_display, strategy_params = prepare_single_ticker_strategy(
            cast(str, ticker),
            start_date,
            end_date,
            strategy_type,
            strategy_params,
            optimise,
        )
        company_name1 = get_full_company_name(ticker_display)

    if data is None or data.is_empty():
        st.write("No data available for the selected ticker and date range")
        return

    # Create a display name for the company or companies
    if company_name2:
        company_display = f"{company_name1} vs. {company_name2}"
    else:
        company_display = company_name1

    # Run the backtest and display the results
    tickers = (
        ticker_display.split(" vs. ")
        if strategy_type == "Pairs Trading"
        else ticker_display
    )
    results, metrics = run_backtest(data, strategy_type, strategy_params, tickers)

    display_performance_metrics(metrics, company_display)
    plot_equity_curve(results, ticker_display, company_display)
    plot_strategy_returns(results, ticker_display, company_display)
    display_returns_by_month(results)

    # Display the raw data from Yahoo Finance for the backtest period
    st.header(f"Raw Data for {company_display}")
    st.dataframe(
        data.to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    # Display historical results
    display_historical_results()


if __name__ == "__main__":
    main()
