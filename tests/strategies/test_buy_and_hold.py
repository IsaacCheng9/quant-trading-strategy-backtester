"""
Tests for the Buy and Hold strategy class.
"""

from datetime import date, timedelta

import polars as pl
from quant_trading_strategy_backtester.strategies.buy_and_hold import BuyAndHoldStrategy


def test_buy_and_hold_strategy_initialisation() -> None:
    params = {}  # Buy and Hold doesn't require parameters
    strategy = BuyAndHoldStrategy(params)
    assert isinstance(strategy, BuyAndHoldStrategy)


def test_buy_and_hold_strategy_generate_signals(mock_polars_data: pl.DataFrame) -> None:
    params = {}
    strategy = BuyAndHoldStrategy(params)
    signals = strategy.generate_signals(mock_polars_data)
    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {"Date", "Close", "signal", "positions"}
    assert all(col in signals.columns for col in EXPECTED_COLS)
    assert (signals["signal"] == 1).all(), "All signals should be 1 (buy)"
    assert (signals["positions"] == 1).all(), "All positions should be 1"


def test_buy_and_hold_strategy_with_mock_data():
    # Create mock data
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    prices = [100 + i for i in range(100)]  # Steadily increasing prices

    mock_polars_data = pl.DataFrame({"Date": dates, "Close": prices})

    params = {}
    strategy = BuyAndHoldStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(mock_polars_data)

    # Check if signals are generated correctly
    assert signals["signal"].sum() == len(signals), "All signals should be buy (1)"
    assert (signals["positions"] == 1).all(), "All positions should be 1"

    # Check if the strategy maintains the buy position throughout
    assert (signals["signal"] == 1).all(), "Buy signal should be maintained throughout"


def test_buy_and_hold_strategy_with_empty_data():
    empty_data = pl.DataFrame(schema=[("Date", pl.Date), ("Close", pl.Float64)])
    params = {}
    strategy = BuyAndHoldStrategy(params)
    signals = strategy.generate_signals(empty_data)

    assert isinstance(signals, pl.DataFrame)
    assert signals.is_empty()
    EXPECTED_COLS = {"Date", "Close", "signal", "positions"}
    assert all(col in signals.columns for col in EXPECTED_COLS)


def test_buy_and_hold_strategy_with_various_price_movements():
    # Create mock data with various price movements
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]

    # Prices that go up, down, and sideways
    prices = (
        [100 + i for i in range(30)]  # Uptrend
        + [130 - i for i in range(30)]  # Downtrend
        + [100] * 40  # Sideways
    )

    mock_polars_data = pl.DataFrame({"Date": dates, "Close": prices})

    params = {}
    strategy = BuyAndHoldStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(mock_polars_data)

    # Check if the strategy maintains the buy position regardless of price movements
    assert (signals["signal"] == 1).all(), "Buy signal should be maintained throughout"
    assert (signals["positions"] == 1).all(), "All positions should be 1"
