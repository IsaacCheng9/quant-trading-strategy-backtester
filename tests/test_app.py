"""
Tests for Streamlit app helpers.
"""

import datetime

import polars as pl

from quant_trading_strategy_backtester.app import _get_benchmark_date_range


def test_get_benchmark_date_range_uses_exclusive_yahoo_end_date() -> None:
    """Verify benchmark loading includes the displayed final backtest date."""
    data = pl.DataFrame(
        {
            "Date": [
                datetime.datetime(2020, 1, 2, 0, 0),
                datetime.datetime(2020, 1, 3, 0, 0),
            ],
            "Close": [100.0, 101.0],
        }
    )

    start_date, end_date = _get_benchmark_date_range(data)

    assert start_date == datetime.date(2020, 1, 2)
    assert end_date == datetime.date(2020, 1, 4)
