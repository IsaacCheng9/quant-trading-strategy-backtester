"""
Calculate benchmark-relative performance metrics.
"""

import math

import numpy as np
import polars as pl

from quant_trading_strategy_backtester.backtester import TRADING_DAYS_PER_YEAR

BENCHMARK_TICKER = "SPY"


def calculate_benchmark_relative_metrics(
    strategy_results: pl.DataFrame,
    benchmark_data: pl.DataFrame,
    risk_free_return_rate_annual: float = 0.0,
) -> dict[str, float]:
    """
    Calculate strategy performance relative to a benchmark return stream.

    Args:
        strategy_results: Backtest results containing Date and strategy_returns.
        benchmark_data: Benchmark price data containing Date and Close.
        risk_free_return_rate_annual: Annualised risk-free return rate.

    Returns:
        Benchmark-relative metrics aligned on common trading dates.

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_columns(strategy_results, {"Date", "strategy_returns"}, "strategy")
    _validate_columns(benchmark_data, {"Date", "Close"}, "benchmark")

    if strategy_results.is_empty() or benchmark_data.is_empty():
        return _undefined_metrics(0)

    strategy_returns_data = (
        strategy_results.select(["Date", "strategy_returns"])
        .sort("Date")
        .with_columns(pl.col("Date").shift(1).alias("strategy_previous_date"))
    )
    benchmark_returns = (
        benchmark_data.select(["Date", "Close"])
        .sort("Date")
        .with_columns(
            [
                pl.col("Date").shift(1).alias("benchmark_previous_date"),
                ((pl.col("Close") / pl.col("Close").shift(1)) - 1)
                .fill_null(0)
                .alias("benchmark_returns"),
            ]
        )
        .select(["Date", "benchmark_previous_date", "benchmark_returns"])
    )
    aligned_returns = (
        strategy_returns_data.join(benchmark_returns, on="Date", how="inner")
        .filter(
            (
                pl.col("strategy_previous_date").is_null()
                & pl.col("benchmark_previous_date").is_null()
            )
            | (pl.col("strategy_previous_date") == pl.col("benchmark_previous_date"))
        )
        .drop_nulls(["strategy_returns", "benchmark_returns"])
        .sort("Date")
    )

    if aligned_returns.height < 2:
        return _undefined_metrics(aligned_returns.height)

    strategy_returns = aligned_returns["strategy_returns"].cast(pl.Float64).to_numpy()
    benchmark_returns_array = (
        aligned_returns["benchmark_returns"].cast(pl.Float64).to_numpy()
    )
    finite_mask = np.isfinite(strategy_returns) & np.isfinite(benchmark_returns_array)
    strategy_returns = strategy_returns[finite_mask]
    benchmark_returns_array = benchmark_returns_array[finite_mask]

    if len(strategy_returns) < 2:
        return _undefined_metrics(len(strategy_returns))

    strategy_total_return = float(np.prod(1 + strategy_returns) - 1)
    benchmark_total_return = float(np.prod(1 + benchmark_returns_array) - 1)
    excess_return = strategy_total_return - benchmark_total_return

    periods = TRADING_DAYS_PER_YEAR
    rf_daily = (1 + risk_free_return_rate_annual) ** (1 / periods) - 1
    strategy_excess_returns = strategy_returns - rf_daily
    benchmark_excess_returns = benchmark_returns_array - rf_daily

    beta = _calculate_beta(strategy_excess_returns, benchmark_excess_returns)
    alpha = (
        float(
            (
                np.mean(strategy_excess_returns)
                - beta * np.mean(benchmark_excess_returns)
            )
            * periods
        )
        if math.isfinite(beta)
        else float("nan")
    )
    information_ratio = _calculate_information_ratio(
        strategy_returns - benchmark_returns_array
    )

    return {
        "Strategy Aligned Total Return": strategy_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "Excess Return": excess_return,
        "Beta": beta,
        "Alpha": alpha,
        "Information Ratio": information_ratio,
        "Benchmark Observations": float(len(strategy_returns)),
    }


def _validate_columns(
    data: pl.DataFrame, required_columns: set[str], label: str
) -> None:
    """Raise if a DataFrame is missing required columns."""
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{label.title()} data missing required columns: {sorted(missing_columns)}"
        )


def _calculate_beta(
    strategy_excess_returns: np.ndarray, benchmark_excess_returns: np.ndarray
) -> float:
    """Calculate strategy beta to the benchmark return stream."""
    benchmark_variance = float(np.var(benchmark_excess_returns, ddof=1))
    if not math.isfinite(benchmark_variance) or np.isclose(benchmark_variance, 0.0):
        return float("nan")

    covariance = float(np.cov(strategy_excess_returns, benchmark_excess_returns)[0, 1])
    return covariance / benchmark_variance


def _calculate_information_ratio(active_returns: np.ndarray) -> float:
    """Calculate annualised information ratio from active returns."""
    active_return_std = float(np.std(active_returns, ddof=1))
    if not math.isfinite(active_return_std) or np.isclose(active_return_std, 0.0):
        return float("nan")

    return float(
        (TRADING_DAYS_PER_YEAR**0.5)
        * float(np.mean(active_returns))
        / active_return_std
    )


def _undefined_metrics(observations: int) -> dict[str, float]:
    """Return undefined benchmark-relative metrics with an observation count."""
    return {
        "Strategy Aligned Total Return": float("nan"),
        "Benchmark Total Return": float("nan"),
        "Excess Return": float("nan"),
        "Beta": float("nan"),
        "Alpha": float("nan"),
        "Information Ratio": float("nan"),
        "Benchmark Observations": float(observations),
    }
