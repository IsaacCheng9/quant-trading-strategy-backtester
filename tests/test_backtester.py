"""
Contains tests for the Backtester class and its methods.
"""

import datetime
import math
from typing import Any

import polars as pl
import pytest
from quant_trading_strategy_backtester.backtester import Backtester, build_trade_ledger
from quant_trading_strategy_backtester.models import StrategyModel
from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategies.buy_and_hold import BuyAndHoldStrategy
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


class MockPairsWeightedStrategy(BaseStrategy):
    """A mock pairs strategy with predetermined leg weights."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.signals = params["signals"]
        self.leg_1_weights = params["leg_1_weights"]
        self.leg_2_weights = params["leg_2_weights"]

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return predetermined pair signals and leg weights."""
        leg_1_weight_changes = _calculate_changes(self.leg_1_weights)
        leg_2_weight_changes = _calculate_changes(self.leg_2_weights)

        return data.select(
            [pl.col("Date"), pl.col("Close_1"), pl.col("Close_2")]
        ).with_columns(
            [
                pl.Series("signal", self.signals),
                pl.Series("leg_1_weight", self.leg_1_weights),
                pl.Series("leg_2_weight", self.leg_2_weights),
                pl.Series("leg_1_weight_change", leg_1_weight_changes),
                pl.Series("leg_2_weight_change", leg_2_weight_changes),
                pl.Series("signal", self.signals)
                .diff()
                .fill_null(0)
                .alias("position_change"),
            ]
        )


def _calculate_changes(values: list[float]) -> list[float]:
    """Return first differences while treating the first value as an entry."""
    return [values[0]] + [
        current - previous for previous, current in zip(values, values[1:])
    ]


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
    EXPECTED_COLS = {
        "position_change",
        "gross_strategy_returns",
        "transaction_costs",
        "strategy_returns",
        "trade_turnover",
        "cumulative_transaction_costs",
        "gross_cumulative_returns",
        "cumulative_returns",
        "equity_curve",
    }
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
    EXPECTED_METRICS = {
        "Total Return",
        "Gross Total Return",
        "Sharpe Ratio",
        "Max Drawdown",
        "Total Costs",
        "Cost Drag",
        "Trade Events",
        "Total Turnover",
    }
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
    assert abs(results["position_change"].sum()) < 1e-6
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
    backtester.save_results()

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
    # position_change = [0, 1, 0, 0, -1]  (signal.diff())
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


def test_transaction_costs_reduce_returns():
    """
    Verify that transaction costs reduce returns proportionally to the
    number and size of position changes.
    """
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

    signals = [0.0, 1.0, 1.0, 1.0, 0.0]
    strategy = MockHoldingStrategy({"signals": signals})

    # Run without costs.
    bt_no_costs = Backtester(data, strategy, transaction_cost_bps=0.0, slippage_bps=0.0)
    results_no_costs = bt_no_costs.run()
    return_no_costs = float(results_no_costs["cumulative_returns"].tail(1).item())

    # Run with costs (10bps transaction + 5bps slippage).
    bt_with_costs = Backtester(
        data, strategy, transaction_cost_bps=10.0, slippage_bps=5.0
    )
    results_with_costs = bt_with_costs.run()
    return_with_costs = float(results_with_costs["cumulative_returns"].tail(1).item())

    assert return_with_costs < return_no_costs, (
        "Returns with transaction costs should be lower"
    )

    # There are 2 position changes (entry on day 2, exit on day 5),
    # each costing 15bps = 0.0015. Verify the cost is deducted on
    # those days only.
    cost_per_trade = 15.0 / 10_000
    returns_no = results_no_costs["strategy_returns"].to_list()
    returns_with = results_with_costs["strategy_returns"].to_list()

    for i in range(len(returns_no)):
        position_change = abs(results_with_costs["position_change"][i])
        expected_diff = position_change * cost_per_trade
        actual_diff = returns_no[i] - returns_with[i]
        assert abs(actual_diff - expected_diff) < 1e-10, (
            f"Day {i + 1}: expected cost deduction {expected_diff}, got {actual_diff}"
        )


def test_backtester_reports_cost_attribution_metrics():
    """
    Verify that performance metrics include gross returns and cost drag.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
                datetime.date(2020, 1, 4),
            ],
            "Close": [100.0, 110.0, 121.0, 133.1],
        }
    )
    signals = [0.0, 1.0, 1.0, 0.0]
    strategy = MockHoldingStrategy({"signals": signals})
    backtester = Backtester(data, strategy, transaction_cost_bps=10.0, slippage_bps=5.0)

    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    assert results["gross_strategy_returns"].to_list() == pytest.approx(
        [0.0, 0.0, 0.1, 0.1]
    )
    assert results["transaction_costs"].to_list() == pytest.approx(
        [0.0, 0.0015, 0.0, 0.0015]
    )
    assert metrics["Gross Total Return"] > metrics["Total Return"]
    assert metrics["Total Costs"] == pytest.approx(0.003)
    assert metrics["Cost Drag"] > 0
    assert metrics["Trade Events"] == 2
    assert metrics["Total Turnover"] == 2


def test_default_transaction_costs_reduce_returns():
    """
    Verify that the default transaction costs (5bps each) produce
    lower returns than explicitly zero costs.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "Close": [100.0, 110.0, 121.0],
        }
    )

    signals = [0.0, 1.0, 1.0]
    strategy = MockHoldingStrategy({"signals": signals})

    bt_default = Backtester(data, strategy)
    bt_zero = Backtester(data, strategy, transaction_cost_bps=0.0, slippage_bps=0.0)

    results_default = bt_default.run()
    results_zero = bt_zero.run()

    default_cum = float(results_default["cumulative_returns"].tail(1).item())
    zero_cum = float(results_zero["cumulative_returns"].tail(1).item())
    assert default_cum < zero_cum, (
        "Default costs should produce lower returns than zero costs"
    )


def test_buy_and_hold_charges_initial_entry_cost_only():
    """
    Verify that buy and hold pays costs on entry rather than on every row.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "Close": [100.0, 110.0, 121.0],
        }
    )
    strategy = BuyAndHoldStrategy({})

    bt_no_costs = Backtester(
        data,
        strategy,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    results_no_costs = bt_no_costs.run()

    bt_with_costs = Backtester(
        data,
        strategy,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
    )
    results_with_costs = bt_with_costs.run()

    cost_per_trade = 8.0 / 10_000
    return_diffs = [
        no_cost - with_cost
        for no_cost, with_cost in zip(
            results_no_costs["strategy_returns"],
            results_with_costs["strategy_returns"],
        )
    ]

    assert return_diffs == [cost_per_trade, 0.0, 0.0]
    assert results_with_costs["position_change"].to_list() == [1.0, 0.0, 0.0]


def test_pairs_returns_use_normalised_leg_weights():
    """
    Verify that pairs returns use the previous day's normalised leg weights.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "Close_1": [100.0, 110.0, 121.0],
            "Close_2": [100.0, 100.0, 100.0],
        }
    )
    strategy = MockPairsWeightedStrategy(
        {
            "signals": [0.0, 1.0, 1.0],
            "leg_1_weights": [0.0, 1 / 3, 1 / 3],
            "leg_2_weights": [0.0, -2 / 3, -2 / 3],
        }
    )

    backtester = Backtester(data, strategy, transaction_cost_bps=0.0, slippage_bps=0.0)
    results = backtester.run()

    assert results["strategy_returns"].to_list() == pytest.approx([0.0, 0.0, 0.1 / 3])


def test_pairs_costs_use_leg_weight_changes():
    """
    Verify that pairs costs are charged on leg turnover, not signal changes.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
                datetime.date(2020, 1, 4),
            ],
            "Close_1": [100.0, 100.0, 100.0, 100.0],
            "Close_2": [100.0, 100.0, 100.0, 100.0],
        }
    )
    strategy = MockPairsWeightedStrategy(
        {
            "signals": [0.0, 1.0, 1.0, 1.0],
            "leg_1_weights": [0.0, 0.5, 1 / 3, 1 / 3],
            "leg_2_weights": [0.0, -0.5, -2 / 3, -2 / 3],
        }
    )

    backtester = Backtester(data, strategy, transaction_cost_bps=10.0, slippage_bps=5.0)
    results = backtester.run()

    cost_rate = 15.0 / 10_000
    assert results["pair_turnover"].to_list() == pytest.approx([0.0, 1.0, 1 / 3, 0.0])
    assert results["trade_turnover"].to_list() == pytest.approx([0.0, 1.0, 1 / 3, 0.0])
    assert results["transaction_costs"].to_list() == pytest.approx(
        [0.0, cost_rate, cost_rate / 3, 0.0]
    )
    assert results["strategy_returns"].to_list() == pytest.approx(
        [0.0, -cost_rate, -(cost_rate / 3), 0.0]
    )
    assert results["position_change"].to_list() == [0.0, 1.0, 0.0, 0.0]


def test_trade_ledger_records_single_asset_entries_and_exits():
    """
    Verify that the trade ledger captures entry, exit, costs, and holding time.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
                datetime.date(2020, 1, 4),
            ],
            "Close": [100.0, 110.0, 121.0, 133.1],
        }
    )
    strategy = MockHoldingStrategy({"signals": [0.0, 1.0, 1.0, 0.0]})
    backtester = Backtester(data, strategy, transaction_cost_bps=10.0, slippage_bps=5.0)
    results = backtester.run()

    ledger = build_trade_ledger(results)

    assert ledger["Action"].to_list() == ["Enter Long", "Exit Long"]
    assert ledger["Reason"].to_list() == [
        "Strategy signal changed",
        "Strategy signal changed",
    ]
    assert ledger["Transaction Costs"].to_list() == pytest.approx([0.0015, 0.0015])
    assert ledger["Holding Period Days"].to_list() == [None, 2]


def test_trade_ledger_normalises_datetime_dates():
    """
    Verify that live datetime dates can be rendered in the trade ledger.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
                datetime.datetime(2020, 1, 3),
            ],
            "Close": [100.0, 110.0, 121.0],
        }
    )
    strategy = MockHoldingStrategy({"signals": [0.0, 1.0, 0.0]})
    backtester = Backtester(data, strategy)
    results = backtester.run()

    ledger = build_trade_ledger(results)

    assert ledger["Date"].dtype == pl.Date
    assert ledger["Date"].to_list() == [
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
    ]


def test_trade_ledger_records_pair_rebalances():
    """
    Verify that pair leg-weight changes are visible even without signal changes.
    """
    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "Close_1": [100.0, 100.0, 100.0],
            "Close_2": [100.0, 100.0, 100.0],
        }
    )
    strategy = MockPairsWeightedStrategy(
        {
            "signals": [0.0, 1.0, 1.0],
            "leg_1_weights": [0.0, 0.5, 1 / 3],
            "leg_2_weights": [0.0, -0.5, -2 / 3],
        }
    )
    backtester = Backtester(data, strategy, transaction_cost_bps=10.0, slippage_bps=5.0)
    results = backtester.run()

    ledger = build_trade_ledger(results)

    assert ledger["Action"].to_list() == ["Enter Long", "Rebalance"]
    assert ledger["Reason"].to_list() == [
        "Strategy signal changed",
        "Leg weights changed",
    ]
    assert ledger["Turnover"].to_list() == pytest.approx([1.0, 1 / 3])
    assert ledger["Leg 1 Weight"].to_list() == pytest.approx([0.5, 1 / 3])
    assert ledger["Leg 2 Weight"].to_list() == pytest.approx([-0.5, -2 / 3])
