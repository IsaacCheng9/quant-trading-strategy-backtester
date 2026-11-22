"""
Contains functions related to optimisation and backtesting for strategy
parameters and ticker pairs.
"""

import datetime
import itertools
import math
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import polars as pl

from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.cointegration import (
    COINTEGRATION_P_VALUE_THRESHOLD,
    evaluate_cointegration,
)
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

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]


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


def get_training_data(
    data: pl.DataFrame, walk_forward: bool = False, n_folds: int = WALK_FORWARD_FOLDS
) -> pl.DataFrame:
    """
    Return the training data used for ticker or parameter optimisation.

    Args:
        data: Historical price data.
        walk_forward: Whether the optimisation used walk-forward validation.
        n_folds: Number of walk-forward test folds.

    Returns:
        The standard training split, or the final walk-forward training
        context.
    """
    if walk_forward:
        segment_size = len(data) // (n_folds + 1)
        if segment_size < 2:
            raise ValueError(
                f"Not enough data for {n_folds} folds: "
                f"{len(data)} rows, need at least {2 * (n_folds + 1)}"
            )
        return data[: segment_size * n_folds]

    train_data, _ = _split_data(data)
    return train_data


def evaluate_validation_period(
    context_data: pl.DataFrame,
    validation_data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    tickers: str | list[str],
    initial_capital: float = 100000.0,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Evaluate validation rows while preserving prior strategy state.

    Run the strategy over all data available through the validation period,
    retain only validation dates, and reset cumulative metrics at the
    validation boundary.

    Args:
        context_data: Price data through the end of the validation period.
        validation_data: Exact rows to score as out of sample.
        strategy_type: The type of strategy to evaluate.
        strategy_params: Scalar parameters for the selected strategy.
        tickers: The ticker or tickers used in the backtest.
        initial_capital: Capital used to reset the validation equity curve.

    Returns:
        Validation-period results and performance metrics.

    Raises:
        ValueError: If validation data is empty or produces no result rows.
    """
    if validation_data.is_empty():
        raise ValueError("Validation data must not be empty")

    context_results, _ = run_backtest(
        context_data,
        strategy_type,
        strategy_params,
        tickers,
    )
    validation_dates = validation_data.select(pl.col("Date").unique())
    validation_results = context_results.join(validation_dates, on="Date", how="inner")
    if validation_results.is_empty():
        raise ValueError("Validation dates produced no backtest results")

    validation_results = _reset_cumulative_result_columns(
        validation_results,
        initial_capital,
    )
    strategy = create_strategy(strategy_type, strategy_params)
    backtester = Backtester(
        validation_data,
        strategy,
        initial_capital=initial_capital,
        tickers=tickers,
    )
    backtester.results = validation_results
    include_initial_period = _has_prior_context(context_data, validation_data)
    metrics = backtester.get_performance_metrics(
        include_initial_period=include_initial_period
    )
    if metrics is None:
        raise ValueError("Validation period produced no performance metrics")

    return validation_results, metrics


def _has_prior_context(
    context_data: pl.DataFrame, validation_data: pl.DataFrame
) -> bool:
    """Return whether context_data starts strictly before validation_data."""
    validation_start = validation_data.select(pl.col("Date").min()).item()
    if validation_start is None:
        return False

    return (
        context_data.filter(
            pl.col("Date") < pl.lit(validation_start, dtype=context_data["Date"].dtype)
        ).height
        > 0
    )


def _reset_cumulative_result_columns(
    results: pl.DataFrame, initial_capital: float
) -> pl.DataFrame:
    """Reset cumulative returns and costs from the first validation row."""
    return results.with_columns(
        [
            (1 + pl.col("gross_strategy_returns"))
            .cum_prod()
            .alias("gross_cumulative_returns"),
            (1 + pl.col("strategy_returns")).cum_prod().alias("cumulative_returns"),
            (initial_capital * (1 + pl.col("strategy_returns")).cum_prod()).alias(
                "equity_curve"
            ),
            pl.col("transaction_costs").cum_sum().alias("cumulative_transaction_costs"),
        ]
    )


def _optimisation_score(
    metrics: dict[str, float], metric_name: str = "Sharpe Ratio"
) -> float | None:
    """Return a finite optimisation score, or None for invalid metrics."""
    score = metrics.get(metric_name)
    if score is None or not math.isfinite(score):
        return None

    return score


def count_valid_parameter_combinations(
    strategy_type: str,
    parameter_ranges: Mapping[str, range | list[int | float] | int | float],
) -> int:
    """Return the number of valid scalar parameter combinations."""
    return len(_valid_parameter_combinations(strategy_type, parameter_ranges))


def count_candidate_pairs(top_companies: list[tuple[str, float]]) -> int:
    """Return pair candidates after same-company filtering."""
    return len(_candidate_pairs(top_companies))


def run_optimisation(
    data: pl.DataFrame,
    strategy_type: str,
    strategy_params: dict[str, Any],
    start_date: datetime.date,
    end_date: datetime.date,
    tickers: str | list[str],
    walk_forward: bool = False,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
    fold_results_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run optimisation for strategy parameters or ticker selection.

    Args:
        data: Historical price data.
        strategy_type: The type of strategy being optimised.
        strategy_params: Initial strategy parameters or parameter ranges.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        tickers: The ticker or tickers used in the backtest.
        walk_forward: Whether to use walk-forward validation.
        progress_callback: Optional callback invoked as
            (current, total, message) during candidate evaluation.
        status_callback: Optional callback invoked with informational status
            messages.
        fold_results_callback: Optional callback invoked with per-fold
            walk-forward results when walk_forward is True.

    Returns:
        A tuple containing:
            - Optimised strategy parameters or selected ticker parameters.
            - Performance metrics for the optimised strategy.
    """
    if strategy_type == "Buy and Hold":
        top_companies = get_top_sp500_companies(NUM_TOP_COMPANIES_ONE_TICKER)
        best_ticker, strategy_params, metrics = optimise_buy_and_hold_ticker(
            top_companies,
            start_date,
            end_date,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        if status_callback is not None:
            status_callback(f"Best ticker for Buy and Hold: {best_ticker}")
    elif walk_forward:
        strategy_params, metrics, fold_results = walk_forward_optimise(
            data,
            strategy_type,
            cast(dict[str, range | list[int | float]], strategy_params),
            tickers,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        if fold_results_callback is not None:
            fold_results_callback(fold_results)
    else:
        train_data, test_data = _split_data(data)
        strategy_params, metrics = optimise_strategy_params(
            train_data,
            strategy_type,
            cast(dict[str, range | list[int | float]], strategy_params),
            tickers,
            test_data=test_data,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

    return strategy_params, metrics


def optimise_buy_and_hold_ticker(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """Optimise ticker selection for the Buy and Hold strategy.

    Use a 70/30 train/test split: rank tickers by total return on training
    data, then re-evaluate the winner on test data.

    Args:
        top_companies: List of tuples containing ticker symbols and market
            caps of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        progress_callback: Optional callback invoked as
            (current, total, message) for each candidate ticker.
        status_callback: Optional callback invoked with summary messages.

    Returns:
        A tuple containing the best ticker, strategy parameters, and
        out-of-sample performance metrics.
    """
    best_ticker = None
    best_test_data = None
    best_total_return = float("-inf")
    evaluated_tickers = 0

    total_tickers = len(top_companies)

    for i, (ticker, _) in enumerate(top_companies):
        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_tickers,
                f"Evaluating ticker {i + 1} / {total_tickers}: {ticker}",
            )

        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        if data is None or data.is_empty():
            continue

        evaluated_tickers += 1
        train_data, test_data = _split_data(data)

        backtester = Backtester(train_data, BuyAndHoldStrategy({}), tickers=ticker)
        backtester.run()
        train_metrics = backtester.get_performance_metrics()

        if train_metrics is None:
            continue

        total_return = _optimisation_score(train_metrics, "Total Return")
        if total_return is not None and total_return > best_total_return:
            best_total_return = total_return
            best_ticker = ticker
            best_test_data = test_data

    if status_callback is not None:
        status_callback(
            f"Ticker selection evaluated {evaluated_tickers} of {total_tickers} "
            "candidates. Ranking used the training split and final metrics use "
            "the held-out test split."
        )

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
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> str:
    """Optimise ticker selection for single-ticker strategies.

    Use a 70/30 train/test split and rank tickers by Sharpe ratio on training
    data.

    Args:
        top_companies: List of tuples containing ticker symbols and market
            caps of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        strategy_type: The type of strategy being used.
        strategy_params: Strategy parameters.
        progress_callback: Optional callback invoked as
            (current, total, message) for each candidate ticker.
        status_callback: Optional callback invoked with summary messages.

    Returns:
        The best ticker.
    """
    best_ticker = None
    best_sharpe_ratio = float("-inf")
    evaluated_tickers = 0

    total_tickers = len(top_companies)

    # Use fixed parameter values for ticker evaluation.
    fixed_params = {
        k: v[0] if isinstance(v, (list, range)) else v
        for k, v in strategy_params.items()
    }

    for i, (ticker, _) in enumerate(top_companies):
        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_tickers,
                f"Evaluating ticker {i + 1} / {total_tickers}: {ticker}",
            )

        data = load_yfinance_data_one_ticker(ticker, start_date, end_date)
        if data is None or data.is_empty():
            continue

        evaluated_tickers += 1
        train_data, _ = _split_data(data)
        _, train_metrics = run_backtest(train_data, strategy_type, fixed_params, ticker)

        score = _optimisation_score(train_metrics)
        if score is not None and score > best_sharpe_ratio:
            best_sharpe_ratio = score
            best_ticker = ticker

    if status_callback is not None:
        status_callback(
            f"Ticker selection evaluated {evaluated_tickers} of {total_tickers} "
            "candidates. Ranking used the training split; displayed results use "
            "the validation split."
        )

    if not best_ticker:
        raise ValueError("Single ticker strategy ticker optimisation failed")

    return best_ticker


def optimise_strategy_params(
    data: pl.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range | list[int | float]],
    tickers: str | list[str],
    test_data: pl.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> tuple[dict[str, int | float], dict[str, float]]:
    """Optimise strategy parameters across all valid grid combinations.

    Run grid search on `data` (training set). If `test_data` is provided,
    evaluate the best parameters on it and return out-of-sample metrics.
    Otherwise, return in-sample metrics.

    Args:
        data: Training price data for grid search.
        strategy_type: The type of strategy to optimise.
        parameter_ranges: A dictionary of parameters and their possible values
            to test.
        tickers: The ticker or tickers used in the backtest.
        test_data: Optional held-out test data for out-of-sample evaluation.
        progress_callback: Optional callback invoked as
            (current, total, message) for each parameter combination.
        status_callback: Optional callback invoked with search context
            messages.

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

    if status_callback is not None:
        if test_data is not None:
            status_callback(
                "Parameter optimisation tests "
                f"{total_combinations} valid combinations on {len(data)} "
                f"training rows. Displayed optimisation metrics use "
                f"{len(test_data)} held-out test rows."
            )
        else:
            status_callback(
                "Parameter optimisation tests "
                f"{total_combinations} valid combinations on {len(data)} "
                "training rows."
            )

    for i, current_params in enumerate(param_combinations):
        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_combinations,
                f"Evaluating parameter combination {i + 1} / {total_combinations}",
            )

        _, metrics = run_backtest(data, strategy_type, current_params, tickers)

        score = _optimisation_score(metrics)
        if score is not None and score > best_sharpe_ratio:
            best_sharpe_ratio = score
            best_params = current_params
            best_metrics = metrics

    if not best_params or not best_metrics:
        raise ValueError("Parameter optimisation failed")

    # Evaluate best params on held-out test data if provided.
    if test_data is not None:
        context_data = pl.concat([data, test_data], how="vertical")
        _, test_metrics = evaluate_validation_period(
            context_data,
            test_data,
            strategy_type,
            best_params,
            tickers,
        )
        return best_params, test_metrics

    return best_params, best_metrics


def optimise_pairs_trading_tickers(
    top_companies: list[tuple[str, float]],
    start_date: datetime.date,
    end_date: datetime.date,
    strategy_params: dict[str, Any],
    optimise: bool,
    cointegration_p_value_threshold: float = COINTEGRATION_P_VALUE_THRESHOLD,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, float]]:
    """Optimise ticker pair selection and parameters for pairs trading.

    Use a 70/30 train/test split and rank pairs by Sharpe ratio on training
    data. When `optimise` is True, fit parameters on training data and
    evaluate the winning pair on test data.

    Args:
        top_companies: List of tuples containing ticker symbols and market
            caps of top companies.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        strategy_params: Strategy parameters or parameter ranges.
        optimise: Whether to optimise the strategy parameters.
        cointegration_p_value_threshold: Maximum Engle-Granger p-value
            accepted for automatic pair selection.
        progress_callback: Optional callback invoked as
            (current, total, message) for each candidate pair.
        status_callback: Optional callback invoked with summary messages.

    Returns:
        A tuple containing the best ticker pair, best parameters, and best
        metrics.
    """
    best_pair = None
    best_params = None
    best_context_data = None
    best_test_data = None
    best_sharpe_ratio = float("-inf")
    evaluated_pairs = 0

    ticker_pairs = _candidate_pairs(top_companies)
    total_combinations = len(ticker_pairs)
    parameter_combinations = 0
    optimisation_parameter_ranges = None
    if optimise:
        optimisation_parameter_ranges = {
            k: [v] if isinstance(v, (int, float)) else v
            for k, v in strategy_params.items()
        }
        parameter_combinations = count_valid_parameter_combinations(
            "Pairs Trading", optimisation_parameter_ranges
        )

    prev_pair_processing_time = 0.0
    rejected_pairs = 0

    for i, (ticker1, ticker2) in enumerate(ticker_pairs):
        start_time = time.time()
        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_combinations,
                f"Evaluating pair {i + 1} / {total_combinations}: "
                f"{ticker1} vs. {ticker2} "
                f"(prev. pair processing time: {prev_pair_processing_time:.4f} seconds)",
            )

        data = load_yfinance_data_two_tickers(ticker1, ticker2, start_date, end_date)
        if data is None or data.is_empty():
            continue

        evaluated_pairs += 1
        train_data, test_data = _split_data(data)
        cointegration_result = evaluate_cointegration(
            train_data,
            p_value_threshold=cointegration_p_value_threshold,
        )
        if not cointegration_result.is_cointegrated:
            rejected_pairs += 1
            end_time = time.time()
            prev_pair_processing_time = end_time - start_time
            continue

        if optimise:
            current_params, train_metrics = optimise_strategy_params(
                train_data,
                "Pairs Trading",
                cast(
                    dict[str, range | list[int | float]],
                    optimisation_parameter_ranges,
                ),
                [ticker1, ticker2],
            )
        else:
            _, train_metrics = run_backtest(
                train_data, "Pairs Trading", strategy_params, [ticker1, ticker2]
            )
            current_params = strategy_params

        score = _optimisation_score(train_metrics)
        if score is not None and score > best_sharpe_ratio:
            best_sharpe_ratio = score
            best_pair = (ticker1, ticker2)
            best_params = current_params
            best_context_data = data
            best_test_data = test_data

        end_time = time.time()
        prev_pair_processing_time = end_time - start_time

    if (
        not best_pair
        or not best_params
        or best_context_data is None
        or best_test_data is None
    ):
        raise ValueError(
            "Pairs trading optimisation failed: no cointegrated pair produced "
            f"a finite Sharpe ratio ({rejected_pairs} pairs rejected by "
            "cointegration filter)"
        )

    cointegrated_pairs = evaluated_pairs - rejected_pairs
    if status_callback is not None:
        status_callback(
            f"Pair selection evaluated {evaluated_pairs} of {total_combinations} "
            f"candidate pairs. The cointegration filter rejected {rejected_pairs}; "
            "ranking used the training split and final metrics use the held-out "
            "test split."
        )
        if optimise:
            status_callback(
                "Pair parameter optimisation tested "
                f"{parameter_combinations} valid combinations per cointegrated "
                f"pair ({cointegrated_pairs * parameter_combinations} training "
                "backtests)."
            )

    _, best_metrics = evaluate_validation_period(
        best_context_data,
        best_test_data,
        "Pairs Trading",
        best_params,
        list(best_pair),
    )

    return best_pair, best_params, best_metrics


def walk_forward_optimise(
    data: pl.DataFrame,
    strategy_type: str,
    parameter_ranges: dict[str, range | list[int | float]],
    tickers: str | list[str],
    n_folds: int = WALK_FORWARD_FOLDS,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> tuple[dict[str, int | float], dict[str, float], list[dict]]:
    """Optimise strategy parameters using walk-forward validation.

    Split data into (n_folds + 1) equal segments. For each fold, use an
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
        progress_callback: Optional callback invoked as
            (current, total, message) for each fold combination evaluation.
        status_callback: Optional callback invoked with search context
            messages.

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

    total_combinations = len(param_combinations)
    total_steps = n_folds * total_combinations
    if status_callback is not None:
        status_callback(
            "Parameter optimisation tests "
            f"{total_combinations} valid combinations across "
            f"{n_folds} expanding walk-forward folds. "
            "Displayed optimisation metrics are aggregated out-of-sample "
            "fold results."
        )

    fold_results: list[dict] = []
    completed_steps = 0

    for fold in range(n_folds):
        train_end = segment_size * (fold + 1)
        test_end = min(segment_size * (fold + 2), n_rows)
        train_data = data[:train_end]
        test_data = data[train_end:test_end]

        # Grid search on training data.
        best_sharpe = float("-inf")
        best_params: dict[str, int | float] | None = None
        for combo_idx, current_params in enumerate(param_combinations):
            if progress_callback is not None:
                completed_steps += 1
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"Fold {fold + 1} / {n_folds}: evaluating combination "
                    f"{combo_idx + 1} / {total_combinations}",
                )
            _, metrics = run_backtest(
                train_data, strategy_type, current_params, tickers
            )
            score = _optimisation_score(metrics)
            if score is not None and score > best_sharpe:
                best_sharpe = score
                best_params = current_params

        if best_params is None:
            raise ValueError(f"Walk-forward optimisation failed on fold {fold + 1}")

        # Evaluate out of sample without cold-starting rolling strategy state.
        _, oos_metrics = evaluate_validation_period(
            data[:test_end],
            test_data,
            strategy_type,
            best_params,
            tickers,
        )

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
    parameter_ranges: Mapping[str, range | list[int | float] | int | float],
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
    param_values = [_parameter_values(value) for value in parameter_ranges.values()]

    return [
        current_params
        for params in itertools.product(*param_values)
        if is_valid_strategy_params(
            strategy_type, current_params := dict(zip(param_names, params))
        )
    ]


def _parameter_values(
    value: range | list[int | float] | float,
) -> list[int | float]:
    """Return parameter candidate values from ranges, lists, or scalars."""
    if isinstance(value, range):
        return list(value)
    if isinstance(value, list):
        return value

    return [value]


def _candidate_pairs(top_companies: list[tuple[str, float]]) -> list[tuple[str, str]]:
    """Return ticker pairs after excluding likely same-company pairs."""
    ticker_pairs = list(
        itertools.combinations([company[0] for company in top_companies], 2)
    )
    return [pair for pair in ticker_pairs if not is_same_company(pair[0], pair[1])]


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
