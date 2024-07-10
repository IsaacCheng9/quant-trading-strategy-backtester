import pandas as pd
from quant_trading_strategy_backtester.strategy_templates import (
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
    assert "signal" in signals.columns
    assert "short_mavg" in signals.columns
    assert "long_mavg" in signals.columns
    assert "positions" in signals.columns
    assert signals["signal"].isin([0.0, 1.0]).all()
