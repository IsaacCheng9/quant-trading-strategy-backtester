"""
Tests for the Pairs Trading strategy class.
"""

from datetime import date, timedelta

import polars as pl
import pytest
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)


def _make_shocked_pair_data(num_days: int = 100) -> pl.DataFrame:
    """Create a pair with a stable beta and temporary residual shocks."""
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    close_2 = [100.0 + i for i in range(num_days)]
    residuals = [0.0] * num_days
    for i in range(50, 60):
        residuals[i] = 20.0
    for i in range(70, 80):
        residuals[i] = -20.0

    close_1 = [10.0 + 2.0 * close_2[i] + residuals[i] for i in range(num_days)]
    return pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})


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
    EXPECTED_COLS = {
        "hedge_ratio",
        "spread",
        "z_score",
        "signal",
        "leg_1_weight",
        "leg_2_weight",
        "leg_1_weight_change",
        "leg_2_weight_change",
        "position_change",
    }
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0, -1.0]).all()


def test_pairs_trading_strategy_initialisation() -> None:
    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    assert strategy.window == 20
    assert strategy.entry_z_score == 2.0
    assert strategy.exit_z_score == 0.5


def test_pairs_trading_strategy_rejects_invalid_thresholds() -> None:
    params = {"window": 20, "entry_z_score": 1.0, "exit_z_score": 1.0}

    with pytest.raises(ValueError, match="exit_z_score must be less"):
        PairsTradingStrategy(params)


def test_pairs_trading_strategy_signal_generation() -> None:
    data = _make_shocked_pair_data()
    params = {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    assert signals.filter(pl.col("signal") == 1.0).height > 0
    assert signals.filter(pl.col("signal") == -1.0).height > 0
    assert signals.filter(pl.col("signal") == 0.0).height > 0


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
    mock_polars_data = _make_shocked_pair_data()

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

    # Check if position changes are calculated correctly
    non_zero_changes = signals.filter(pl.col("position_change") != 0)
    assert len(non_zero_changes) > 0, "No position changes"
    assert signals["position_change"].abs().sum() > 0, "No position changes"

    # Check if the spread and z-score are calculated correctly.
    valid_spreads = signals.filter(pl.col("hedge_ratio").is_not_null())
    spread_error = (
        valid_spreads["spread"]
        - valid_spreads["Close_1"]
        + valid_spreads["hedge_ratio"] * valid_spreads["Close_2"]
    )
    max_spread_error = max(abs(float(error)) for error in spread_error.to_list())
    assert max_spread_error < 1e-10, "Spread calculation is incorrect"
    assert signals["z_score"].null_count() == 0, "Z-score contains null values"


def test_pairs_trading_strategy_estimates_hedge_ratio() -> None:
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(40)]
    close_2 = [100.0 + i for i in range(40)]
    close_1 = [15.0 + 2.0 * price for price in close_2]
    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})

    strategy = PairsTradingStrategy(
        {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5}
    )
    signals = strategy.generate_signals(data)
    valid_signals = signals.filter(pl.col("hedge_ratio").is_not_null())

    assert valid_signals["hedge_ratio"].to_list() == pytest.approx([2.0] * 31)
    assert valid_signals["spread"].to_list() == pytest.approx([15.0] * 31)


def test_pairs_trading_strategy_is_invariant_to_second_leg_price_scale() -> None:
    data = _make_shocked_pair_data(num_days=80)
    scaled_data = data.with_columns((pl.col("Close_2") * 10.0).alias("Close_2"))
    params = {"window": 10, "entry_z_score": 1.5, "exit_z_score": 0.5}

    base_signals = PairsTradingStrategy(params).generate_signals(data)
    scaled_signals = PairsTradingStrategy(params).generate_signals(scaled_data)

    assert base_signals["spread"].to_list() == pytest.approx(
        scaled_signals["spread"].to_list(), nan_ok=True
    )
    assert base_signals["z_score"].to_list() == pytest.approx(
        scaled_signals["z_score"].to_list(), nan_ok=True
    )
