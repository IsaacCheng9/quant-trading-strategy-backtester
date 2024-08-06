"""
Contains tests for trading strategy templates.
"""

from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest
from quant_trading_strategy_backtester.strategies.base import Strategy
from quant_trading_strategy_backtester.strategies.mean_reversion import (
    MeanReversionStrategy,
)
from quant_trading_strategy_backtester.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)


def test_moving_average_crossover_strategy_initialization() -> None:
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
    assert (
        signals.filter(pl.col("Date") == crossover_index)["signal"].item() == 1.0
    ), "Buy signal not generated at crossover"

    # Check if the strategy generates a sell signal when short MA crosses below long MA
    crossunder_indices = signals.filter(
        pl.col("short_mavg") < pl.col("long_mavg")
    ).select("Date")
    assert len(crossunder_indices) > 0, "No crossunder detected"
    crossunder_index = crossunder_indices.row(-1)[0]
    assert (
        signals.filter(pl.col("Date") == crossunder_index)["signal"].item() == 0.0
    ), "Sell signal not generated at crossunder"

    # Check if positions are calculated correctly
    non_zero_positions = signals.filter(pl.col("positions") != 0)
    assert len(non_zero_positions) > 0, "No position changes"
    assert signals["positions"].abs().sum() > 0, "No position changes"


def test_mean_reversion_strategy_initialization() -> None:
    params = {"window": 20, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    assert strategy.window == 20
    assert strategy.std_dev == 2.0


def test_mean_reversion_strategy_generate_signals(
    mock_polars_data: pl.DataFrame,
) -> None:
    params = {"window": 5, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    signals = strategy.generate_signals(mock_polars_data)
    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {"signal", "mean", "std", "upper_band", "lower_band", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0, -1.0]).all()


@pytest.mark.parametrize(
    "strategy_class,params",
    [
        (MovingAverageCrossoverStrategy, {"short_window": 5, "long_window": 20}),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
        ),
    ],
)
def test_strategy_with_empty_data(
    strategy_class: Strategy, params: dict[str, Any]
) -> None:
    empty_data = pl.DataFrame(
        schema=[("Close", pl.Float64)]
        if strategy_class != PairsTradingStrategy
        else [("Close_1", pl.Float64), ("Close_2", pl.Float64)]
    )
    strategy = strategy_class(params)  # type: ignore
    signals = strategy.generate_signals(empty_data)

    assert isinstance(signals, pl.DataFrame)
    assert "Date" in signals.columns
    assert "signal" in signals.columns
    assert "positions" in signals.columns
    assert signals.is_empty()
    assert signals.is_empty()


def test_pairs_trading_strategy_initialization() -> None:
    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    assert strategy.window == 20
    assert strategy.entry_z_score == 2.0
    assert strategy.exit_z_score == 0.5


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
