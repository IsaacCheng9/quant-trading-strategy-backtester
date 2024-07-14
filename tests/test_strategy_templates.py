import pandas as pd
import pytest
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
)


def test_moving_average_crossover_strategy_initialization():
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    assert strategy.short_window == 5
    assert strategy.long_window == 20


def test_moving_average_crossover_strategy_generate_signals(mock_data):
    params = {"short_window": 5, "long_window": 20}
    strategy = MovingAverageCrossoverStrategy(params)
    signals = strategy.generate_signals(mock_data)
    assert isinstance(signals, pd.DataFrame)
    EXPECTED_COLS = {"signal", "short_mavg", "long_mavg", "positions"}
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].isin([0.0, 1.0]).all()


def test_mean_reversion_strategy_initialization():
    params = {"window": 20, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    assert strategy.window == 20
    assert strategy.std_dev == 2.0


def test_mean_reversion_strategy_generate_signals(mock_data):
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
    ],
)
def test_strategy_with_empty_data(strategy_class, params):
    empty_data = pd.DataFrame(columns=["Close"])
    strategy = strategy_class(params)
    signals = strategy.generate_signals(empty_data)
    assert signals.empty
