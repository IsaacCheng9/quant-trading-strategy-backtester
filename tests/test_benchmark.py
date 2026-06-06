"""
Tests for benchmark-relative reporting.
"""

import datetime
import math

import numpy as np
import polars as pl
import pytest

from quant_trading_strategy_backtester.backtester import TRADING_DAYS_PER_YEAR
from quant_trading_strategy_backtester.benchmark import (
    calculate_benchmark_relative_metrics,
)


def test_calculate_benchmark_relative_metrics() -> None:
    """Verify benchmark-relative metrics are calculated from aligned returns."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    strategy_returns = np.array([0.0, 0.02, -0.01, 0.03])
    benchmark_returns = np.array([0.0, 0.01, 0.0, 0.02])
    strategy_results = pl.DataFrame(
        {"Date": dates, "strategy_returns": strategy_returns}
    )
    benchmark_data = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 101.0, 101.0, 103.02],
        }
    )

    metrics = calculate_benchmark_relative_metrics(strategy_results, benchmark_data)

    strategy_total_return = float(np.prod(1 + strategy_returns) - 1)
    benchmark_total_return = float(np.prod(1 + benchmark_returns) - 1)
    beta = float(np.cov(strategy_returns, benchmark_returns)[0, 1]) / float(
        np.var(benchmark_returns, ddof=1)
    )
    alpha = float(
        (np.mean(strategy_returns) - beta * np.mean(benchmark_returns))
        * TRADING_DAYS_PER_YEAR
    )
    active_returns = strategy_returns - benchmark_returns
    information_ratio = float(
        (TRADING_DAYS_PER_YEAR**0.5)
        * float(np.mean(active_returns))
        / float(np.std(active_returns, ddof=1))
    )

    assert metrics["Strategy Aligned Total Return"] == pytest.approx(
        strategy_total_return
    )
    assert metrics["Benchmark Total Return"] == pytest.approx(benchmark_total_return)
    assert metrics["Excess Return"] == pytest.approx(
        strategy_total_return - benchmark_total_return
    )
    assert metrics["Beta"] == pytest.approx(beta)
    assert metrics["Alpha"] == pytest.approx(alpha)
    assert metrics["Information Ratio"] == pytest.approx(information_ratio)
    assert metrics["Benchmark Observations"] == 4


def test_calculate_benchmark_relative_metrics_drops_mismatched_return_periods() -> None:
    """Verify benchmark metrics ignore mismatched return intervals."""
    strategy_results = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "strategy_returns": [0.0, 0.02, 0.03],
        }
    )
    benchmark_data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 3),
            ],
            "Close": [100.0, 101.0],
        }
    )

    metrics = calculate_benchmark_relative_metrics(strategy_results, benchmark_data)

    assert math.isnan(metrics["Strategy Aligned Total Return"])
    assert math.isnan(metrics["Benchmark Total Return"])
    assert metrics["Benchmark Observations"] == 1


def test_calculate_benchmark_relative_metrics_handles_constant_benchmark() -> None:
    """Verify undefined relative metrics are displayed as NaN when needed."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    strategy_results = pl.DataFrame(
        {"Date": dates, "strategy_returns": [0.0, 0.01, -0.005]}
    )
    benchmark_data = pl.DataFrame({"Date": dates, "Close": [100.0, 100.0, 100.0]})

    metrics = calculate_benchmark_relative_metrics(strategy_results, benchmark_data)

    assert math.isnan(metrics["Beta"])
    assert math.isnan(metrics["Alpha"])
    assert math.isfinite(metrics["Information Ratio"])
    assert metrics["Benchmark Observations"] == 3


def test_calculate_benchmark_relative_metrics_requires_columns() -> None:
    """Verify missing strategy or benchmark columns fail clearly."""
    data = pl.DataFrame({"Date": [datetime.date(2020, 1, 1)]})

    with pytest.raises(ValueError, match="Strategy"):
        calculate_benchmark_relative_metrics(data, data.with_columns(pl.lit(1.0)))

    with pytest.raises(ValueError, match="Benchmark"):
        calculate_benchmark_relative_metrics(
            data.with_columns(pl.lit(0.0).alias("strategy_returns")),
            data,
        )
