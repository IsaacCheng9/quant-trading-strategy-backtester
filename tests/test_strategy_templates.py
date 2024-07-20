from typing import Any

import numpy as np
import pandas as pd
import pytest
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
    Strategy,
)


def test_moving_average_crossover_strategy_initialization() -> None:
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    assert strategy.short_window == 5
    assert strategy.long_window == 20


def test_moving_average_crossover_strategy_generate_signals(
    mock_data: pd.DataFrame,
) -> None:
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    signals = strategy.generate_signals(mock_data)
    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"signal", "short_mavg", "long_mavg", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0]).all()


def test_mean_reversion_strategy_initialization() -> None:
    params = {"window": 20, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    assert strategy.window == 20
    assert strategy.std_dev == 2.0


def test_mean_reversion_strategy_generate_signals(mock_data: pd.DataFrame) -> None:
    params = {"window": 5, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    signals = strategy.generate_signals(mock_data)
    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"signal", "mean", "std", "upper_band", "lower_band", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0, -1.0]).all()


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
    empty_data = pd.DataFrame(
        columns=["Close"]
        if strategy_class != PairsTradingStrategy
        else ["Close_1", "Close_2"]
    )
    strategy = strategy_class(params)  # type: ignore
    signals = strategy.generate_signals(empty_data)
    assert signals.empty


def test_pairs_trading_strategy_initialization() -> None:
    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    assert strategy.window == 20
    assert strategy.entry_z_score == 2.0
    assert strategy.exit_z_score == 0.5


def test_pairs_trading_strategy_generate_signals() -> None:
    # Create mock data for two assets
    dates = pd.date_range(start="2020-01-01", periods=100)
    data = pd.DataFrame(
        {
            "Close_1": np.random.randn(100).cumsum() + 100,
            "Close_2": np.random.randn(100).cumsum() + 100,
        },
        index=dates,
    )

    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"spread", "z_score", "signal", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0, -1.0]).all()


def test_pairs_trading_strategy_signal_generation() -> None:
    dates = pd.date_range(start="2020-01-01", periods=100)
    data = pd.DataFrame(
        {"Close_1": [100] * 50 + [110] * 50, "Close_2": [100] * 50 + [100] * 50},
        index=dates,
    )

    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    # Check if the strategy generates the expected signals
    # Should go short asset 1, long asset 2
    assert signals["signal"].iloc[50] == -1.0
    # Should maintain position after entry
    assert (signals["signal"].iloc[51:] != 0.0).any()  #


def test_pairs_trading_strategy_with_missing_data() -> None:
    data = pd.DataFrame(
        {"Close_1": [100, 101, 102, 103], "Close_2": [100, 101, np.nan, 102]}
    )
    params = {"window": 2, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    # One NaN signal due to missing data
    assert signals["signal"].isna().sum() == 1
    # The third signal should be NaN
    assert np.isnan(signals["signal"].iloc[2])
    # The fourth signal should not be NaN
    assert not np.isnan(signals["signal"].iloc[3])


def test_pairs_trading_strategy_with_invalid_data() -> None:
    data = pd.DataFrame(
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
