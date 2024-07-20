from typing import Any
import pandas as pd
import pytest
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategy_templates import (
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
    Strategy,
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
def test_backtester_initialization(
    request: pytest.FixtureRequest,
    strategy_class: Strategy,
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(data, strategy)
    assert backtester.data is data
    assert isinstance(backtester.strategy, strategy_class)  # type: ignore
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
def test_backtester_run(
    request: pytest.FixtureRequest,
    strategy_class: Strategy,
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)  # type: ignore
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
    request: pytest.FixtureRequest,
    strategy_class: Strategy,
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric in metrics


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
def test_backtester_with_invalid_data(strategy_class, params) -> None:
    if strategy_class == PairsTradingStrategy:
        invalid_data = pd.DataFrame({"Invalid_1": [1, 2, 3], "Invalid_2": [4, 5, 6]})
    else:
        invalid_data = pd.DataFrame({"Invalid": [1, 2, 3]})

    strategy = strategy_class(params)
    backtester = Backtester(invalid_data, strategy)

    with pytest.raises((KeyError, ValueError)):
        backtester.run()


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
def test_backtester_with_insufficient_data_all_strategies(
    strategy_class: Strategy, params: dict[str, Any]
) -> None:
    if strategy_class == PairsTradingStrategy:
        insufficient_data = pd.DataFrame(
            {"Close_1": [100, 101], "Close_2": [100, 102]},
            index=pd.date_range(start="2020-01-01", periods=2),
        )
    else:
        insufficient_data = pd.DataFrame(
            {"Close": [100, 101]}, index=pd.date_range(start="2020-01-01", periods=2)
        )

    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(insufficient_data, strategy)
    results = backtester.run()

    # Check that no meaningful trading occurred
    assert (
        abs(results["positions"].sum()) < 1e-6
    )  # Allow for small floating-point errors

    # Check that the equity curve doesn't change significantly
    assert abs(results["equity_curve"].iloc[-1] - backtester.initial_capital) < 1e-6

    # Check that cumulative returns are close to 1 (no significant change)
    assert abs(results["cumulative_returns"].iloc[-1] - 1) < 1e-6

    # Verify that the DataFrame has the expected number of rows
    assert len(results) == len(insufficient_data)
