from datetime import date, timedelta

import polars as pl
import pytest
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)


def test_pairs_trading_strategy_generate_signals() -> None:
    # Create mock data for two assets
    start_date = date(2020, 1, 1)
    end_date = date(2020, 4, 9)
    num_days = (end_date - start_date).days + 1

    date_range = [start_date + timedelta(days=i) for i in range(num_days)]

    data = pl.DataFrame(
        {
            "Date": pl.Series(date_range),
            "Close_1": pl.arange(0, num_days, eager=True).cum_sum() + 100,
            "Close_2": pl.arange(0, num_days, eager=True).cum_sum() * 0.5 + 100,
        }
    )

    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {"spread", "z_score", "signal", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0, -1.0]).all()


def test_pairs_trading_strategy_initialisation() -> None:
    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    assert strategy.window == 20
    assert strategy.entry_z_score == 2.0
    assert strategy.exit_z_score == 0.5


def test_pairs_trading_strategy_signal_generation() -> None:
    start_date = date(2020, 1, 1)
    end_date = date(2020, 4, 9)
    num_days = (end_date - start_date).days + 1

    date_range = [start_date + timedelta(days=i) for i in range(num_days)]

    data = pl.DataFrame(
        {
            "Date": pl.Series(date_range),
            "Close_1": [100] * 50 + [110] * (num_days - 50),
            "Close_2": [100] * num_days,
        }
    )

    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    # Check if the strategy generates the expected signals
    # Should go short asset 1, long asset 2
    assert signals["signal"][50] == -1.0
    # Should maintain position after entry
    assert (signals["signal"][51:] != 0.0).any()


def test_pairs_trading_strategy_with_invalid_data() -> None:
    data = pl.DataFrame(
        {
            "Close_1": [100, 101, 102],
            "Close_3": [100, 101, 102],  # Invalid column name
        }
    )
    params = {"window": 2, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    with pytest.raises(
        ValueError, match="Data must contain 'Close_1' and 'Close_2' columns"
    ):
        strategy.generate_signals(data)


def test_pairs_trading_strategy_with_mock_polars_data():
    """Test the Pairs Trading strategy with mock data."""
    # Create mock data
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]

    # Create price series for two assets with both divergence and convergence
    # Increase then decrease
    prices1 = [100 + i for i in range(50)] + [150 - i for i in range(50)]
    # Steadily increasing
    prices2 = [100 + i * 0.1 for i in range(100)]

    mock_polars_data = pl.DataFrame(
        {"Date": dates, "Close_1": prices1, "Close_2": prices2}
    )

    # Strategy parameters
    params = {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.5}

    strategy = PairsTradingStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(mock_polars_data)

    # Check if signals are generated correctly
    assert signals["signal"].null_count() == 0, "Null signals generated"

    long_signals = signals.filter(pl.col("signal") == 1)
    short_signals = signals.filter(pl.col("signal") == -1)
    exit_signals = signals.filter(pl.col("signal") == 0)
    assert len(long_signals) > 0, "No long signals (1) generated"
    assert len(short_signals) > 0, "No short signals (-1) generated"
    assert len(exit_signals) > 0, "No exit signals (0) generated"

    # Check if the strategy generates a signal when z-score crosses thresholds
    entry_long = signals.filter(
        (pl.col("z_score") < -params["entry_z_score"]) & (pl.col("signal") == 1)
    )
    entry_short = signals.filter(
        (pl.col("z_score") > params["entry_z_score"]) & (pl.col("signal") == -1)
    )
    exit_positions = signals.filter(
        (pl.col("z_score").abs() < params["exit_z_score"]) & (pl.col("signal") == 0)
    )
    assert len(entry_long) > 0, "No long entry signals generated"
    assert len(entry_short) > 0, "No short entry signals generated"
    assert len(exit_positions) > 0, "No exit signals generated"

    # Check if positions are calculated correctly
    non_zero_positions = signals.filter(pl.col("positions") != 0)
    assert len(non_zero_positions) > 0, "No position changes"
    assert signals["positions"].abs().sum() > 0, "No position changes"

    # Check if the spread and z-score are calculated correctly
    assert (
        signals["spread"] == signals["Close_1"] - signals["Close_2"]
    ).all(), "Spread calculation is incorrect"
    assert signals["z_score"].null_count() == 0, "Z-score contains null values"
