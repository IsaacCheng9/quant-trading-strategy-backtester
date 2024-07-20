import pandas as pd
import pytest
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
)


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_pairs_data",
        ),
    ],
)
def test_backtester_initialization(request, strategy_class, params, data_fixture):
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
    backtester = Backtester(data, strategy)
    assert backtester.data is data
    assert isinstance(backtester.strategy, strategy_class)
    assert backtester.initial_capital == 100000.0


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_pairs_data",
        ),
    ],
)
def test_backtester_run(request, strategy_class, params, data_fixture):
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
    backtester = Backtester(data, strategy)
    results = backtester.run()
    assert isinstance(results, pd.DataFrame)
    EXPECTED_COLS = {"positions", "strategy_returns", "equity_curve"}
    for col in EXPECTED_COLS:
        assert col in results.columns


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_pairs_data",
        ),
    ],
)
def test_backtester_get_performance_metrics(
    request, strategy_class, params, data_fixture
):
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric in metrics


def test_backtester_with_invalid_data():
    invalid_data = pd.DataFrame({"Invalid": [1, 2, 3]})
    strategy = MovingAverageCrossoverStrategy({"short_window": 5, "long_window": 20})
    backtester = Backtester(invalid_data, strategy)
    with pytest.raises(KeyError):
        backtester.run()


def test_backtester_with_insufficient_data():
    insufficient_data = pd.DataFrame({"Close": [100, 101]})
    strategy = MovingAverageCrossoverStrategy({"short_window": 5, "long_window": 20})
    backtester = Backtester(insufficient_data, strategy)
    results = backtester.run()
    assert results.empty
