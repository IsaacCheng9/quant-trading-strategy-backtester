import pandas as pd
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MovingAverageCrossoverStrategy,
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
    EXPECTED_COLS = {"positions", "strategy_returns", "equity_curve"}
    for col in EXPECTED_COLS:
        assert col in results.columns


def test_backtester_get_performance_metrics(mock_data):
    strategy = MovingAverageCrossoverStrategy(5, 20)
    backtester = Backtester(mock_data, strategy)
    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric
