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

    # Cold-start row 0 has no prior return period and is excluded.
    aligned_strategy = strategy_returns[1:]
    aligned_benchmark = benchmark_returns[1:]
    strategy_total_return = float(np.prod(1 + aligned_strategy) - 1)
    benchmark_total_return = float(np.prod(1 + aligned_benchmark) - 1)
    beta = float(np.cov(aligned_strategy, aligned_benchmark)[0, 1]) / float(
        np.var(aligned_benchmark, ddof=1)
    )
    alpha = float(
        (np.mean(aligned_strategy) - beta * np.mean(aligned_benchmark))
        * TRADING_DAYS_PER_YEAR
    )
    active_returns = aligned_strategy - aligned_benchmark
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
    assert metrics["Benchmark Observations"] == 3


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
    assert metrics["Benchmark Observations"] == 0


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
    assert metrics["Benchmark Observations"] == 2


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


def test_calculate_benchmark_relative_metrics_with_prior_date_captures_day_zero() -> (
    None
):
    """Verify prior_date aligns real day-0 returns on pl.Date columns."""
    prior_date = datetime.date(2020, 1, 1)
    validation_dates = [
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
    ]
    strategy_returns = np.array([0.04, -0.01, 0.02])
    benchmark_closes = [100.0, 105.0, 102.9, 104.958]
    benchmark_returns = np.array([0.05, -0.02, 0.02])

    strategy_results = pl.DataFrame(
        {"Date": validation_dates, "strategy_returns": strategy_returns}
    )
    benchmark_data = pl.DataFrame(
        {
            "Date": [prior_date, *validation_dates],
            "Close": benchmark_closes,
        }
    )

    metrics = calculate_benchmark_relative_metrics(
        strategy_results,
        benchmark_data,
        prior_date=prior_date,
    )

    expected_strategy_total = float(np.prod(1 + strategy_returns) - 1)
    expected_benchmark_total = float(np.prod(1 + benchmark_returns) - 1)
    expected_beta = float(np.cov(strategy_returns, benchmark_returns)[0, 1]) / float(
        np.var(benchmark_returns, ddof=1)
    )

    assert metrics["Benchmark Observations"] == 3
    assert metrics["Strategy Aligned Total Return"] == pytest.approx(
        expected_strategy_total
    )
    assert metrics["Benchmark Total Return"] == pytest.approx(expected_benchmark_total)
    assert metrics["Excess Return"] == pytest.approx(
        expected_strategy_total - expected_benchmark_total
    )
    assert metrics["Beta"] == pytest.approx(expected_beta)


def test_calculate_benchmark_relative_metrics_with_prior_date_timezone_datetime() -> (
    None
):
    """Verify prior_date works on timezone-aware pl.Datetime Date columns."""
    prior_dt = datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.UTC)
    validation_dts = [
        datetime.datetime(2020, 1, 2, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2020, 1, 3, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2020, 1, 4, 0, 0, tzinfo=datetime.UTC),
    ]
    strategy_results = pl.DataFrame(
        {
            "Date": validation_dts,
            "strategy_returns": [0.02, -0.01, 0.03],
        }
    )
    benchmark_data = pl.DataFrame(
        {
            "Date": [prior_dt, *validation_dts],
            "Close": [100.0, 102.0, 100.98, 104.0094],
        }
    )

    # Pass both datetime and plain date forms of prior_date.
    metrics_from_dt = calculate_benchmark_relative_metrics(
        strategy_results,
        benchmark_data,
        prior_date=prior_dt,
    )
    metrics_from_date = calculate_benchmark_relative_metrics(
        strategy_results,
        benchmark_data,
        prior_date=prior_dt.date(),
    )

    expected_benchmark_total = 104.0094 / 100.0 - 1
    assert metrics_from_dt["Benchmark Observations"] == 3
    assert metrics_from_dt["Benchmark Total Return"] == pytest.approx(
        expected_benchmark_total
    )
    assert metrics_from_dt["Beta"] == pytest.approx(1.0)
    assert metrics_from_date["Benchmark Observations"] == 3
    assert metrics_from_date["Benchmark Total Return"] == pytest.approx(
        expected_benchmark_total
    )


def test_calculate_benchmark_relative_metrics_drops_day_zero_if_prior_missing() -> None:
    """Verify row 0 is dropped when benchmark lacks prior_date."""
    prior_date = datetime.date(2020, 1, 1)
    validation_dates = [
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
    ]
    strategy_results = pl.DataFrame(
        {
            "Date": validation_dates,
            "strategy_returns": [0.05, 0.02, -0.01],
        }
    )
    # Benchmark starts on validation day 0 without prior_date.
    benchmark_data = pl.DataFrame(
        {
            "Date": validation_dates,
            "Close": [100.0, 101.0, 100.0],
        }
    )

    metrics = calculate_benchmark_relative_metrics(
        strategy_results,
        benchmark_data,
        prior_date=prior_date,
    )

    # Only days 1 and 2 have matching prior dates; day 0 is dropped.
    expected_strategy_total = (1.02 * 0.99) - 1
    expected_benchmark_total = (100.0 / 100.0) - 1
    assert metrics["Benchmark Observations"] == 2
    assert metrics["Strategy Aligned Total Return"] == pytest.approx(
        expected_strategy_total
    )
    assert metrics["Benchmark Total Return"] == pytest.approx(expected_benchmark_total)


def test_calculate_benchmark_relative_metrics_excludes_cold_start_vs_prior_date() -> (
    None
):
    """Verify cold-start row 0 is excluded unless prior_date is provided."""
    prior_date = datetime.date(2020, 1, 1)
    dates = [
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
        datetime.date(2020, 1, 5),
    ]
    # Row 0 has a distinct return (-0.05) to make inclusion vs exclusion clear.
    strategy_returns = np.array([-0.05, 0.02, -0.01, 0.03])
    strategy_results = pl.DataFrame(
        {"Date": dates, "strategy_returns": strategy_returns}
    )
    cold_benchmark = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 101.0, 100.495, 103.50985],
        }
    )
    validation_benchmark = pl.DataFrame(
        {
            "Date": [prior_date, *dates],
            "Close": [100.0, 98.0, 98.98, 98.4851, 101.439653],
        }
    )

    cold_metrics = calculate_benchmark_relative_metrics(
        strategy_results, cold_benchmark
    )
    validation_metrics = calculate_benchmark_relative_metrics(
        strategy_results, validation_benchmark, prior_date=prior_date
    )

    expected_cold_strategy = float(np.prod(1 + strategy_returns[1:]) - 1)
    expected_validation_strategy = float(np.prod(1 + strategy_returns) - 1)

    assert cold_metrics["Benchmark Observations"] == 3
    assert cold_metrics["Strategy Aligned Total Return"] == pytest.approx(
        expected_cold_strategy
    )
    assert validation_metrics["Benchmark Observations"] == 4
    assert validation_metrics["Strategy Aligned Total Return"] == pytest.approx(
        expected_validation_strategy
    )
    assert cold_metrics["Beta"] != pytest.approx(validation_metrics["Beta"])
    assert cold_metrics["Alpha"] != pytest.approx(validation_metrics["Alpha"])
    assert cold_metrics["Information Ratio"] != pytest.approx(
        validation_metrics["Information Ratio"]
    )
