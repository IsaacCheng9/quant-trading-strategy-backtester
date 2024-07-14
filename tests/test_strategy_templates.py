import pandas as pd
import pytest
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
)


def test_moving_average_crossover_strategy_initialization():
    strategy = MovingAverageCrossoverStrategy(5, 20)
    assert strategy.short_window == 5
    assert strategy.long_window == 20


def test_moving_average_crossover_strategy_generate_signals(mock_data):
    strategy = MovingAverageCrossoverStrategy(5, 20)
    signals = strategy.generate_signals(mock_data)
    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"signal", "short_mavg", "long_mavg", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0]).all()


def test_mean_reversion_strategy_initialization():
    strategy = MeanReversionStrategy(20, 2.0)
    assert strategy.window == 20
    assert strategy.std_dev == 2.0


def test_mean_reversion_strategy_generate_signals(mock_data):
    strategy = MeanReversionStrategy(5, 2.0)
    signals = strategy.generate_signals(mock_data)
    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"signal", "mean", "std", "upper_band", "lower_band", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0, -1.0]).all()


@pytest.mark.parametrize(
    "strategy_class", [MovingAverageCrossoverStrategy, MeanReversionStrategy]
)
def test_strategy_with_empty_data(strategy_class):
    empty_data = pd.DataFrame(columns=["Close"])
    strategy = (
        strategy_class(5, 20)
        if strategy_class == MovingAverageCrossoverStrategy
        else strategy_class(5, 2.0)
    )
    signals = strategy.generate_signals(empty_data)
    assert signals.empty
