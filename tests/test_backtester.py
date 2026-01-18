"""
Contains tests for the Backtester class and its methods.
"""

import datetime
import math
from typing import Any

import polars as pl
import pytest
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.models import StrategyModel
from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategies.mean_reversion import (
    MeanReversionStrategy,
)
from quant_trading_strategy_backtester.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)
from conftest import MockHoldingStrategy


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_polars_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_polars_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_polars_pairs_data",
        ),
    ],
)
def test_backtester_initialisation(
    request: pytest.FixtureRequest,
    strategy_class: BaseStrategy,
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(data, strategy)

    # Compare DataFrames
    assert backtester.data.shape == data.shape
    for col in data.columns:
        assert (backtester.data[col] == data[col]).all()

    assert isinstance(backtester.strategy, strategy_class)  # type: ignore
    assert backtester.initial_capital == 100000.0


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_polars_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_polars_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_polars_pairs_data",
        ),
    ],
)
def test_backtester_run(
    request: pytest.FixtureRequest,
    strategy_class: BaseStrategy,
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(data, strategy)
    results = backtester.run()
    assert isinstance(results, pl.DataFrame)
    EXPECTED_COLS = {"positions", "strategy_returns", "equity_curve"}
    for col in EXPECTED_COLS:
        assert col in results.columns


@pytest.mark.parametrize(
    "strategy_class,params,data_fixture",
    [
        (
            MovingAverageCrossoverStrategy,
            {"short_window": 5, "long_window": 20},
            "mock_polars_data",
        ),
        (MeanReversionStrategy, {"window": 5, "std_dev": 2.0}, "mock_polars_data"),
        (
            PairsTradingStrategy,
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            "mock_polars_pairs_data",
        ),
    ],
)
def test_backtester_get_performance_metrics(
    request: pytest.FixtureRequest,
    strategy_class: BaseStrategy,
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
def test_backtester_with_invalid_data(
    strategy_class: BaseStrategy, params: dict[str, Any]
) -> None:
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    if strategy_class == PairsTradingStrategy:
        invalid_data = pl.DataFrame(
            {
                "Date": dates,
                "Close_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                # Missing Close_2 column
            }
        )
    else:
        invalid_data = pl.DataFrame(
            {
                "Date": dates,
                # Missing Close column
                "Open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(invalid_data, strategy)

    with pytest.raises((KeyError, ValueError, pl.exceptions.ColumnNotFoundError)):
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
    strategy_class: BaseStrategy, params: dict[str, Any]
) -> None:
    dates = [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]
    if strategy_class == PairsTradingStrategy:
        insufficient_data = pl.DataFrame(
            {
                "Date": dates,
                "Close_1": [100, 101],
                "Close_2": [100, 102],
            }
        )
    else:
        insufficient_data = pl.DataFrame(
            {
                "Date": dates,
                "Close": [100, 101],
            }
        )

    strategy = strategy_class(params)  # type: ignore
    backtester = Backtester(insufficient_data, strategy)
    results = backtester.run()

    # Check that no meaningful trading occurred
    # Allow for small floating-point errors
    assert abs(results["positions"].sum()) < 1e-6
    # Check that the equity curve doesn't change significantly
    assert abs(results["equity_curve"].tail(1)[0] - backtester.initial_capital) < 1e-6
    # Check that cumulative returns are close to 1 (no significant change)
    assert abs(results["cumulative_returns"].tail(1)[0] - 1) < 1e-6
    # Verify that the DataFrame has the expected number of rows
    assert len(results) == len(insufficient_data)


def test_backtester_save_results(mock_db_session, mock_polars_data):
    strategy = MovingAverageCrossoverStrategy({"short_window": 5, "long_window": 20})
    backtester = Backtester(mock_polars_data, strategy, session=mock_db_session)
    backtester.run()

    # Print metrics for debugging
    metrics = backtester.get_performance_metrics()
    print("Metrics before saving:", metrics)

    # Verify that the results were saved to the mocked database
    saved_strategy = (
        mock_db_session.query(StrategyModel)
        .filter_by(name="MovingAverageCrossoverStrategy")
        .first()
    )

    assert saved_strategy is not None
    print("Saved strategy:", saved_strategy.__dict__)
    assert saved_strategy.parameters == '{"short_window": 5, "long_window": 20}'
    assert saved_strategy.total_return is not None

    # Check if sharpe_ratio is either NaN or None
    assert saved_strategy.sharpe_ratio is None or math.isnan(
        saved_strategy.sharpe_ratio
    )

    assert saved_strategy.max_drawdown is not None


def test_returns_captured_while_holding_position():
    """
    Verifies that returns are captured for all days while holding a position,
    not just on the day after entry.
    """
    # Create data with known daily returns:
    # Day 1 -> 2: +10%, Day 2 -> 3: +10%, Day 3 -> 4: +10%, Day 4 -> 5: -10%
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
                datetime.date(2020, 1, 4),
                datetime.date(2020, 1, 5),
            ],
            "Close": [100.0, 110.0, 121.0, 133.1, 119.79],
        }
    )

    # Signal: flat, then long for 3 days, then flat
    # signal = [0, 1, 1, 1, 0]
    # positions = [0, 1, 0, 0, -1]  (signal.diff())
    signals = [0.0, 1.0, 1.0, 1.0, 0.0]
    strategy = MockHoldingStrategy({"signals": signals})
    backtester = Backtester(data, strategy)
    results = backtester.run()

    # Expected strategy returns (using signal.shift(1)):
    # Day 1: signal.shift(1) = null -> 0, return = 0
    # Day 2: signal.shift(1) = 0, return = 0 (not in position yesterday)
    # Day 3: signal.shift(1) = 1, return = +10% (in position yesterday)
    # Day 4: signal.shift(1) = 1, return = +10% (in position yesterday)
    # Day 5: signal.shift(1) = 1, return = -10% (in position yesterday)
    strategy_returns = results["strategy_returns"].to_list()

    # Days 3-5 should capture ~10% return
    assert abs(strategy_returns[2] - 0.10) < 0.001, (
        f"Day 3 should capture ~10% return, got {strategy_returns[2]}"
    )
    assert abs(strategy_returns[3] - 0.10) < 0.001, (
        f"Day 4 should capture ~10% return, got {strategy_returns[3]}"
    )
    assert abs(strategy_returns[4] - (-0.10)) < 0.01, (
        f"Day 5 should capture ~-10% return, got {strategy_returns[4]}"
    )

    # Cumulative return should be ~(1.1 * 1.1 * 0.9) - 1 = 8.9%
    total_return = float(results["cumulative_returns"].tail(1).item()) - 1
    expected_return = (1.10 * 1.10 * 0.90) - 1  # ~8.9%
    assert abs(total_return - expected_return) < 0.01, (
        f"Total return should be ~{expected_return:.1%}, got {total_return:.1%}"
    )
