"""
Tests for the Moving Average Crossover strategy class.
"""

from datetime import date, timedelta

import polars as pl
from quant_trading_strategy_backtester.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


def test_moving_average_crossover_strategy_initialisation() -> None:
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    assert strategy.short_window == 5
    assert strategy.long_window == 20


def test_moving_average_crossover_strategy_generate_signals(
    mock_polars_data: pl.DataFrame,
) -> None:
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    signals = strategy.generate_signals(mock_polars_data)
    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {"signal", "short_mavg", "long_mavg", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0]).all()


def test_moving_average_crossover_strategy_with_mock_polars_data():
    # Create mock data
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]

    # Create a price series with a clear trend change
    prices = (
        [100 + i for i in range(50)]  # Uptrend
        + [150 - i for i in range(50)]  # Downtrend
    )

    mock_polars_data = pl.DataFrame({"Date": dates, "Close": prices})

    # Strategy parameters
    params = {"short_window": 10, "long_window": 30}

    strategy = MovingAverageCrossoverStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(mock_polars_data)

    # Check if signals are generated correctly
    assert signals["signal"].sum() > 0, "No buy signals generated"
    assert (
        signals["signal"].value_counts().filter(pl.col("signal") == 1.0)["count"][0] > 0
    ), "No buy signals (1) generated"
    assert (
        signals["signal"].value_counts().filter(pl.col("signal") == 0.0)["count"][0] > 0
    ), "No sell signals (0) generated"

    # Check if the strategy generates a buy signal when short MA crosses above long MA
    crossover_indices = signals.filter(
        pl.col("short_mavg") > pl.col("long_mavg")
    ).select("Date")
    assert len(crossover_indices) > 0, "No crossover detected"
    crossover_index = crossover_indices.row(0)[0]
    assert signals.filter(pl.col("Date") == crossover_index)["signal"].item() == 1.0, (
        "Buy signal not generated at crossover"
    )

    # Check if the strategy generates a sell signal when short MA crosses below long MA
    crossunder_indices = signals.filter(
        pl.col("short_mavg") < pl.col("long_mavg")
    ).select("Date")
    assert len(crossunder_indices) > 0, "No crossunder detected"
    crossunder_index = crossunder_indices.row(-1)[0]
    assert signals.filter(pl.col("Date") == crossunder_index)["signal"].item() == 0.0, (
        "Sell signal not generated at crossunder"
    )

    # Check if positions are calculated correctly
    non_zero_positions = signals.filter(pl.col("positions") != 0)
    assert len(non_zero_positions) > 0, "No position changes"
    assert signals["positions"].abs().sum() > 0, "No position changes"
