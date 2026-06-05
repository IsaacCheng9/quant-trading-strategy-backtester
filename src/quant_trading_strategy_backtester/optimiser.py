"""
Contains functions related to optimisation and backtesting for strategy
parameters and ticker pairs.
"""

import datetime
import itertools
import time
from typing import Any, cast

import polars as pl
import streamlit as st

from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.data import (
    get_top_sp500_companies,
    is_same_company,
    load_yfinance_data_one_ticker,
    load_yfinance_data_two_tickers,
)
from quant_trading_strategy_backtester.strategies.base import (
    TRADING_STRATEGIES,
    BaseStrategy,
)
from quant_trading_strategy_backtester.strategies.buy_and_hold import BuyAndHoldStrategy
from quant_trading_strategy_backtester.strategies.mean_reversion import (
    MeanReversionStrategy,
)
from quant_trading_strategy_backtester.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)
from quant_trading_strategy_backtester.strategy_params import (
    is_valid_strategy_params,
)
from quant_trading_strategy_backtester.utils import NUM_TOP_COMPANIES_ONE_TICKER

TRAIN_RATIO = 0.7
WALK_FORWARD_FOLDS = 5


def _split_data(
    data: pl.DataFrame, train_ratio: float = TRAIN_RATIO
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into training and test sets by row count.

    Args:
        data: The full dataset to split.
        train_ratio: Fraction of rows to use for training (default 0.7).

    Returns:
        A tuple of (train_data, test_data).
    """
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def get_validation_data(
    data: pl.DataFrame, walk_forward: bool = False, n_folds: int = WALK_FORWARD_FOLDS
) -> pl.DataFrame:
    """
    Return the held-out data used for final validation display.

    Args:
        data: Historical price data.
        walk_forward: Whether the optimisation used walk-forward validation.
        n_folds: Number of walk-forward test folds.

    Returns:
        The test split for standard optimisation, or the final test fold for
        walk-forward optimisation.
    """
    if walk_forward:
        segment_size = len(data) // (n_folds + 1)
        if segment_size < 2:
            raise ValueError(
                f"Not enough data for {n_folds} folds: "
                f"{len(data)} rows, need at least {2 * (n_folds + 1)}"
            )
        return data[segment_size * n_folds : segment_size * (n_folds + 1)]

    _, test_data = _split_data(data)
    return test_data


def run_optimisation(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    start_date: datetime.date,
    end_date: datetime.date,
    tickers: str | list[str],
    walk_forward: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Runs the optimisation process for strategy parameters or ticker selection.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy being optimised.
        strategy_params: Initial strategy parameters or parameter ranges.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        tickers: The ticker or tickers used in the backtest.
        walk_forward: Whether to use walk-forward validation.

    Returns:
        A tuple containing:
            - Optimised strategy parameters or selected ticker.
            - Performance metrics for the optimised strategy.
    """
    st.info("Optimising strategy. This may take a while...")
    start_time = time.time()

    if strategy_type == "Buy and Hold":
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)
        best_ticker, strategy_params, metrics = optimise_buy_and_hold_ticker(
            top_companies, start_date, end_date
        )
        st.success(f"Best ticker for Buy and Hold: {best_ticker}")
    elif walk_forward:
        strategy_params, metrics, fold_results = walk_forward_optimise(
            data,
            strategy_type,
            cast(dict[str, range | list[int | float]], strategy_params),
            tickers,
        )
        _display_walk_forward_results(fold_results)
    else:
        train_data, test_data = _split_data(data)
        strategy_params, metrics = optimise_strategy_params(
            train_data,
            strategy_type,
            cast(dict[str, range | list[int | float]], strategy_params),
            tickers,
            test_data=test_data,
        )

    end_time = time.time()
    duration = end_time - start_time
    st.success(f"Optimisation complete! Time taken: {duration:.4f} seconds")

    st.header("Optimal Parameters")
    st.write(strategy_params)

    return strategy_params, metrics


def _display_walk_forward_results(fold_results: list[dict]) -> None:
    """
    Display per-fold walk-forward validation results in the Streamlit
    UI.

    Args:
        fold_results: List of per-fold result dictionaries from
            walk_forward_optimise().
    """
    with st.expander("Walk-Forward Validation Details"):
        for fold in fold_results:
            st.subheader(f"Fold {fold['fold']}")
            st.write(
                f"Train: {fold['train_rows']} rows, Test: {fold['test_rows']} rows"
            )
            st.write(f"Best params: {fold['params']}")
            col1, col2 = st.columns(2)
            col1.metric(
                "In-Sample Sharpe",
                f"{fold['in_sample_sharpe']:.4f}",
            )
            col2.metric(
                "Out-of-Sample Sharpe",
                f"{fold['oos_metrics']['Sharpe Ratio']:.4f}",
            )


def optimise_buy_and_hold_ticker(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """
    Optimises ticker selection for the Buy and Hold strategy.

    Uses a 70/30 train/test split: tickers are ranked by total return on
    training data, and the winner is re-evaluated on test data.

    Args:
        top_companies: List of tuples containing ticker symbols and market caps
                       of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.

    Returns:
        A tuple containing the best ticker, strategy parameters, and
        out-of-sample performance metrics.
    """
    best_ticker = None
    best_test_data = None
    best_total_return = float("-inf")

    total_tickers = len(top_companies)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (ticker, _) in enumerate(top_companies):
        status_text.text(f"Evaluating ticker {i + 1} / {total_tickers}: {ticker}")
        progress_bar.progress((i + 1) / total_tickers)

        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        if data is None or data.is_empty():
            continue

        train_data, test_data = _split_data(data)

        backtester = Backtester(train_data, BuyAndHoldStrategy({}), tickers=ticker)
        backtester.run()
        train_metrics = backtester.get_performance_metrics()

        if train_metrics and train_metrics["Total Return"] > best_total_return:
            best_total_return = train_metrics["Total Return"]
            best_ticker = ticker
            best_test_data = test_data

    progress_bar.empty()
    status_text.empty()

    if not best_ticker or best_test_data is None:
        raise ValueError("Buy and Hold optimisation failed")

    backtester = Backtester(best_test_data, BuyAndHoldStrategy({}), tickers=best_ticker)
    backtester.run()
    best_test_metrics = backtester.get_performance_metrics()
    if best_test_metrics is None:
        raise ValueError("Buy and Hold optimisation failed")

    return best_ticker, {}, best_test_metrics


def optimise_single_ticker_strategy_ticker(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_type: str,
    strategy_params: dict[str, Any],
) -> str:
    """
    Optimises ticker selection for single ticker strategies.

    Uses a 70/30 train/test split: tickers are ranked by Sharpe ratio on
    training data.

    Args:
        top_companies: List of tuples containing ticker symbols and market caps
                       of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Strategy parameters.

    Returns:
        The best ticker.
    """
    best_ticker = None
    best_sharpe_ratio = float("-inf")

    total_tickers = len(top_companies)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Use fixed parameter values for ticker evaluation
    fixed_params = {
        k: v[0] if isinstance(v, (list, range)) else v
        for k, v in strategy_params.items()
    }

    for i, (ticker, _) in enumerate(top_companies):
        status_text.text(f"Evaluating ticker {i + 1} / {total_tickers}: {ticker}")
        progress_bar.progress((i + 1) / total_tickers)

        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        if data is None or data.is_empty():
            continue

        train_data, _ = _split_data(data)
        _, train_metrics = run_backtest(train_data, strategy_type, fixed_params, ticker)

        if train_metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = train_metrics["Sharpe Ratio"]
            best_ticker = ticker

    progress_bar.empty()
    status_text.empty()

    if not best_ticker:
        raise ValueError("Single ticker strategy ticker optimisation failed")

    return best_ticker


def optimise_strategy_params(
    data: pl.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range | list[int | float]],
    tickers: str | list[str],
    test_data: pl.DataFrame | None = None,
) -> tuple[dict[str, int | float], dict[str, float]]:
    """
    Optimises strategy parameters by testing all combinations within given
    ranges.

    Grid search runs on `data` (training set). If `test_data` is provided,
    the best parameters are evaluated on it and out-of-sample metrics are
    returned. Otherwise, in-sample metrics are returned.

    Args:
        data: Training price data for grid search.
        strategy_type: The type of strategy to optimise.
        parameter_ranges: A dictionary of parameters and their possible values
                          to test.
        tickers: The ticker or tickers used in the backtest.
        test_data: Optional held-out test data for out-of-sample evaluation.

    Returns:
        A tuple containing the best parameters and their performance metrics.
    """
    best_params = None
    best_metrics = None
    best_sharpe_ratio = float("-inf")

    param_combinations = _valid_parameter_combinations(strategy_type, parameter_ranges)
    total_combinations = len(param_combinations)
    if total_combinations == 0:
        raise ValueError(f"No valid parameter combinations for {strategy_type}")

    # Display progress bar and status text, as this process may take a while.
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, current_params in enumerate(param_combinations):
        status_text.text(
            f"Evaluating parameter combination {i + 1} / {total_combinations}"
        )
        progress_bar.progress((i + 1) / total_combinations)

        _, metrics = run_backtest(data, strategy_type, current_params, tickers)

        if metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = metrics["Sharpe Ratio"]
            best_params = current_params
            best_metrics = metrics

    progress_bar.empty()
    status_text.empty()
    if not best_params or not best_metrics:
        raise ValueError("Parameter optimisation failed")

    # Evaluate best params on held-out test data if provided.
    if test_data is not None:
        _, test_metrics = run_backtest(test_data, strategy_type, best_params, tickers)
        return best_params, test_metrics

    return best_params, best_metrics


def optimise_pairs_trading_tickers(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, float]]:
    """
    Optimises ticker pair selection and strategy parameters for pairs trading.

    Uses a 70/30 train/test split: pairs are ranked by Sharpe ratio on
    training data. When `optimise` is True, parameter fitting also runs
    on training data and the winner is evaluated on test data.

    Args:
        top_companies: List of tuples containing ticker symbols and market caps
                       of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        strategy_params: Strategy parameters or parameter ranges.
        optimise: Whether to optimise the strategy parameters.

    Returns:
        A tuple containing the best ticker pair, best parameters, and best
        metrics.
    """
    best_pair = None
    best_params = None
    best_test_data = None
    best_sharpe_ratio = float("-inf")

    ticker_pairs = list(
        itertools.combinations([company[0] for company in top_companies], 2)
    )
    # Filter out pairs that likely represent the same company
    ticker_pairs = [
        pair for pair in ticker_pairs if not is_same_company(pair[0], pair[1])
    ]
    total_combinations = len(ticker_pairs)
    # Display progress bar and status text, as this process may take a while.
    progress_bar = st.progress(0)
    status_text = st.empty()
    prev_pair_processing_time = 0.0

    for i, (ticker1, ticker2) in enumerate(ticker_pairs):
        start_time = time.time()
        status_text.text(
            f"Evaluating pair {i + 1} / {total_combinations}: {ticker1} vs. {ticker2} "
            f"(prev. pair processing time: {prev_pair_processing_time:.4f} seconds)"
        )
        progress_bar.progress((i + 1) / total_combinations)

        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        if data is None or data.is_empty():
            continue

        train_data, test_data = _split_data(data)

        if optimise:
            # Convert single values to lists for optimisation
            param_ranges = {
                k: [v] if isinstance(v, (int, float)) else v
                for k, v in strategy_params.items()
            }
            current_params, train_metrics = optimise_strategy_params(
                train_data,
                "Pairs Trading",
                param_ranges,
                [ticker1, ticker2],
            )
        else:
            _, train_metrics = run_backtest(
                train_data, "Pairs Trading", strategy_params, [ticker1, ticker2]
            )
            current_params = strategy_params

        if train_metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = train_metrics["Sharpe Ratio"]
            best_pair = (ticker1, ticker2)
            best_params = current_params
            best_test_data = test_data

        end_time = time.time()
        prev_pair_processing_time = end_time - start_time

    progress_bar.empty()
    status_text.empty()
    if not best_pair or not best_params or best_test_data is None:
        raise ValueError("Pairs trading optimisation failed")

    _, best_metrics = run_backtest(
        best_test_data, "Pairs Trading", best_params, list(best_pair)
    )

    return best_pair, best_params, best_metrics


def walk_forward_optimise(
    data: pl.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range | list[int | float]],
    tickers: str | list[str],
    n_folds: int = WALK_FORWARD_FOLDS,
) -> tuple[dict[str, int | float], dict[str, float], list[dict]]:
    """
    Optimises strategy parameters using walk-forward validation.

    Splits data into (n_folds + 1) equal segments. For each fold, we use an
    expanding training window for grid search and evaluate the best parameters
    on the next segment. This tests parameter stability across time rather than
    depending on a single split point.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy to optimise.
        parameter_ranges: A dictionary of parameters and their
            possible values to test.
        tickers: The ticker or tickers used in the backtest.
        n_folds: Number of out-of-sample test folds.

    Returns:
        A tuple containing:
            - Best parameters from the final fold.
            - Aggregated out-of-sample metrics (mean across folds).
            - Per-fold results with params and metrics.
    """
    n_rows = len(data)
    segment_size = n_rows // (n_folds + 1)
    if segment_size < 2:
        raise ValueError(
            f"Not enough data for {n_folds} folds: "
            f"{n_rows} rows, need at least {2 * (n_folds + 1)}"
        )

    param_combinations = _valid_parameter_combinations(strategy_type, parameter_ranges)
    if not param_combinations:
        raise ValueError(f"No valid parameter combinations for {strategy_type}")

    fold_results: list[dict] = []

    for fold in range(n_folds):
        train_end = segment_size * (fold + 1)
        test_end = min(segment_size * (fold + 2), n_rows)
        train_data = data[:train_end]
        test_data = data[train_end:test_end]

        # Grid search on training data.
        best_sharpe = float("-inf")
        best_params: dict[str, int | float] | None = None
        for current_params in param_combinations:
            _, metrics = run_backtest(
                train_data, strategy_type, current_params, tickers
            )
            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_params = current_params

        if best_params is None:
            raise ValueError(f"Walk-forward optimisation failed on fold {fold + 1}")

        # Evaluate best params on out-of-sample test data.
        _, oos_metrics = run_backtest(test_data, strategy_type, best_params, tickers)

        fold_results.append(
            {
                "fold": fold + 1,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "params": best_params,
                "in_sample_sharpe": best_sharpe,
                "oos_metrics": oos_metrics,
            }
        )

    # Aggregate out-of-sample metrics across folds.
    metric_keys = fold_results[0]["oos_metrics"].keys()
    aggregated_metrics = {
        key: sum(f["oos_metrics"][key] for f in fold_results) / n_folds
        for key in metric_keys
    }

    # Return best params from the final fold (largest training set).
    final_params = fold_results[-1]["params"]

    return final_params, aggregated_metrics, fold_results


def _valid_parameter_combinations(
    strategy_type: str,
    parameter_ranges: dict[str, range | list[int | float]],
) -> list[dict[str, int | float]]:
    """
    Return scalar parameter combinations that pass strategy validation.

    Args:
        strategy_type: The strategy being optimised.
        parameter_ranges: Candidate values for each parameter.

    Returns:
        Valid scalar parameter dictionaries for grid search.
    """
    param_names = list(parameter_ranges.keys())
    param_values = [
        list(value) if isinstance(value, range) else value
        for value in parameter_ranges.values()
    ]

    return [
        current_params
        for params in itertools.product(*param_values)
        if is_valid_strategy_params(
            strategy_type, current_params := dict(zip(param_names, params))
        )
    ]


def run_backtest(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    tickers: str | list[str],
) -> tuple[pl.DataFrame, dict]:
    """
    Executes the backtest using the selected strategy and parameters.

    Args:
        data: Historical stock data.
        strategy_type: The type of strategy to use for the backtest.
        strategy_params: Additional parameters required for the strategy.
        tickers: The ticker or tickers used in the backtest.

    Returns:
        A tuple containing the backtest results DataFrame and performance metrics.
    """
    strategy = create_strategy(strategy_type, strategy_params)
    backtester = Backtester(data, strategy, tickers=tickers)
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None, (
        "No results available for the selected ticker and date range"
    )

    return results, metrics


def create_strategy(
    strategy_type: str, strategy_params: dict[str, Any]
) -> BaseStrategy:
    """
    Creates a trading strategy object based on the selected strategy type.

    Args:
        strategy_type: The type of trading strategy.
        strategy_params: A dictionary containing the strategy parameters.
    """
    if strategy_type not in TRADING_STRATEGIES:
        raise ValueError("Invalid strategy type")

    match strategy_type:
        case "Buy and Hold":
            return BuyAndHoldStrategy(strategy_params)
        case "Moving Average Crossover":
            return MovingAverageCrossoverStrategy(strategy_params)
        case "Mean Reversion":
            return MeanReversionStrategy(strategy_params)
        case "Pairs Trading":
            return PairsTradingStrategy(strategy_params)
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")
