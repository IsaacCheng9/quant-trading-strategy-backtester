import pandas as pd
import pytest
from quant_trading_strategy_backtester.strategy_templates import (
    MovingAverageCrossoverStrategy,
)


@pytest.fixture
def mock_data():
    dates = pd.date_range(start="1/1/2020", end="1/31/2020")
    return pd.DataFrame(
        {
            "Open": [100] * len(dates),
            "High": [110] * len(dates),
            "Low": [90] * len(dates),
            "Close": range(100, 100 + len(dates)),
            "Volume": [1000000] * len(dates),
        },
        index=dates,
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
