import pandas as pd
import pytest
from quant_trading_strategy_backtester.backtester import Backtester
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
            "Close": [105] * len(dates),
            "Volume": [1000000] * len(dates),
        },
        index=dates,
    )


def test_backtester_initialization(mock_data):
    strategy = MovingAverageCrossoverStrategy(5, 20)
    backtester = Backtester(mock_data, strategy)
    assert backtester.data is mock_data
    assert isinstance(backtester.strategy, MovingAverageCrossoverStrategy)
    assert backtester.initial_capital == 100000.0


def test_backtester_run(mock_data):
    strategy = MovingAverageCrossoverStrategy(5, 20)
    backtester = Backtester(mock_data, strategy)
    results = backtester.run()
    assert isinstance(results, pd.DataFrame)
    assert "positions" in results.columns
    assert "strategy_returns" in results.columns
    assert "equity_curve" in results.columns


def test_backtester_get_performance_metrics(mock_data):
    strategy = MovingAverageCrossoverStrategy(5, 20)
    backtester = Backtester(mock_data, strategy)
    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
