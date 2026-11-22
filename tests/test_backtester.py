"""
Contains tests for the Backtester class and its methods.
"""

import datetime
import math
from itertools import pairwise
from typing import Any

import polars as pl
import pytest
from conftest import MockHoldingStrategy

from quant_trading_strategy_backtester.backtester import (
    TRADING_DAYS_PER_YEAR,
    Backtester,
    build_trade_ledger,
)
from quant_trading_strategy_backtester.models import StrategyModel
from quant_trading_strategy_backtester.storage import InMemoryStrategyStore
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
    return [values[0]] + [current - previous for previous, current in pairwise(values)]


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
    strategy_class: type[BaseStrategy],
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
    backtester = Backtester(data, strategy)

    # Compare DataFrames
    assert backtester.data.shape == data.shape
    for col in data.columns:
        assert (backtester.data[col] == data[col]).all()

    assert isinstance(backtester.strategy, strategy_class)
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
    strategy_class: type[BaseStrategy],
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
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
    strategy_class: type[BaseStrategy],
    params: dict[str, Any],
    data_fixture: str,
) -> None:
    data = request.getfixturevalue(data_fixture)
    strategy = strategy_class(params)
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {
        "Total Return",
        "Gross Total Return",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max Drawdown",
        "Max Drawdown Duration",
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
    strategy_class: type[BaseStrategy], params: dict[str, Any]
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

    strategy = strategy_class(params)
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
    strategy_class: type[BaseStrategy], params: dict[str, Any]
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

    strategy = strategy_class(params)
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


def test_backtester_save_results(sqlite_store, session_factory, mock_polars_data):
    """Verify Backtester.save_results persists structured strategy results."""
    strategy = MovingAverageCrossoverStrategy({"short_window": 5, "long_window": 20})
    backtester = Backtester(mock_polars_data, strategy, tickers="AAPL")
    backtester.run()

    saved = backtester.save_results(store=sqlite_store)

    assert saved is True
    with session_factory() as session:
        saved_strategy = (
            session.query(StrategyModel)
            .filter_by(name="MovingAverageCrossoverStrategy")
            .first()
        )

    assert saved_strategy is not None
    assert saved_strategy.parameters == {"long_window": 20, "short_window": 5}
    assert saved_strategy.tickers == "AAPL"
    assert saved_strategy.total_return is not None
    assert saved_strategy.sharpe_ratio is None
    assert saved_strategy.max_drawdown is not None


def test_build_strategy_record_handles_unrun_empty_and_date_dtypes() -> None:
    """Verify build_strategy_record guards empty states and date dtypes."""
    empty_data = pl.DataFrame(schema={"Date": pl.Date, "Close": pl.Float64})
    unrun_bt = Backtester(
        pl.DataFrame(
            {
                "Date": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
                "Close": [100.0, 101.0],
            }
        ),
        BuyAndHoldStrategy({}),
        tickers="AAPL",
    )
    with pytest.raises(ValueError, match="hasn't been run"):
        unrun_bt.build_strategy_record()

    empty_bt = Backtester(empty_data, BuyAndHoldStrategy({}), tickers="AAPL")
    empty_bt.run()
    assert empty_bt.get_performance_metrics() is None
    with pytest.raises(ValueError, match="no results"):
        empty_bt.build_strategy_record()

    # Non-empty results with empty input data must raise clearly.
    unrun_bt.run()
    unrun_bt.data = empty_data
    with pytest.raises(ValueError, match="data is empty"):
        unrun_bt.build_strategy_record()

    # Verify both pl.Date and pl.Datetime columns extract start/end dates.
    date_bt = Backtester(
        pl.DataFrame(
            {
                "Date": [datetime.date(2020, 1, 2), datetime.date(2020, 1, 5)],
                "Close": [100.0, 105.0],
            }
        ),
        BuyAndHoldStrategy({}),
        tickers="AAPL",
    )
    date_bt.run()
    date_record = date_bt.build_strategy_record()
    assert date_record.start_date == datetime.date(2020, 1, 2)
    assert date_record.end_date == datetime.date(2020, 1, 5)

    dt_bt = Backtester(
        pl.DataFrame(
            {
                "Date": [
                    datetime.datetime(2020, 2, 3, 15, 0, tzinfo=datetime.UTC),
                    datetime.datetime(2020, 2, 6, 15, 0, tzinfo=datetime.UTC),
                ],
                "Close": [100.0, 102.0],
            }
        ),
        BuyAndHoldStrategy({}),
        tickers="MSFT",
    )
    dt_bt.run()
    dt_record = dt_bt.build_strategy_record()
    assert dt_record.start_date == datetime.date(2020, 2, 3)
    assert dt_record.end_date == datetime.date(2020, 2, 6)


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


def test_backtester_reports_downside_and_drawdown_risk_metrics() -> None:
    """
    Verify downside risk, drawdown depth, and drawdown duration metrics.
    """
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(6)]
    strategy_returns = [0.0, -0.1, -1 / 9, 0.375, -1 / 11, 0.2]
    equity_curve = [100_000.0, 90_000.0, 80_000.0, 110_000.0, 100_000.0, 120_000.0]
    cumulative_returns = [equity / 100_000.0 for equity in equity_curve]
    results = pl.DataFrame(
        {
            "Date": dates,
            "signal": [1.0] * 6,
            "position_change": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gross_strategy_returns": strategy_returns,
            "strategy_returns": strategy_returns,
            "transaction_costs": [0.0] * 6,
            "trade_turnover": [0.0] * 6,
            "gross_cumulative_returns": cumulative_returns,
            "cumulative_returns": cumulative_returns,
            "cumulative_transaction_costs": [0.0] * 6,
            "equity_curve": equity_curve,
        }
    )
    backtester = Backtester(
        pl.DataFrame({"Date": dates, "Close": [100.0] * 6}),
        BuyAndHoldStrategy({}),
    )
    backtester.results = results

    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    risk_returns = strategy_returns[1:]
    downside_deviation = math.sqrt(
        sum(min(return_value, 0.0) ** 2 for return_value in risk_returns)
        / len(risk_returns)
    )
    expected_sortino = (
        (TRADING_DAYS_PER_YEAR**0.5)
        * (sum(risk_returns) / len(risk_returns))
        / downside_deviation
    )
    expected_calmar = (
        1.2 ** (TRADING_DAYS_PER_YEAR / (len(strategy_returns) - 1)) - 1
    ) / 0.2

    assert metrics["Sortino Ratio"] == pytest.approx(expected_sortino)
    assert metrics["Calmar Ratio"] == pytest.approx(expected_calmar)
    assert metrics["Max Drawdown"] == pytest.approx(-0.2)
    assert metrics["Max Drawdown Duration"] == 2


def test_get_performance_metrics_include_initial_period_calmar() -> None:
    """Verify include_initial_period uses all result rows for Calmar."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    strategy_returns = [0.05, -0.1, 0.2, 0.1]
    cumulative_returns = []
    running = 1.0
    for ret in strategy_returns:
        running *= 1 + ret
        cumulative_returns.append(running)
    equity_curve = [100_000.0 * value for value in cumulative_returns]

    results = pl.DataFrame(
        {
            "Date": dates,
            "signal": [1.0] * 4,
            "position_change": [0.0] * 4,
            "gross_strategy_returns": strategy_returns,
            "strategy_returns": strategy_returns,
            "transaction_costs": [0.0] * 4,
            "trade_turnover": [0.0] * 4,
            "gross_cumulative_returns": cumulative_returns,
            "cumulative_returns": cumulative_returns,
            "cumulative_transaction_costs": [0.0] * 4,
            "equity_curve": equity_curve,
        }
    )
    backtester = Backtester(
        pl.DataFrame({"Date": dates, "Close": [100.0] * 4}),
        BuyAndHoldStrategy({}),
    )
    backtester.results = results

    cold_start_metrics = backtester.get_performance_metrics(
        include_initial_period=False
    )
    validation_metrics = backtester.get_performance_metrics(include_initial_period=True)
    assert cold_start_metrics is not None
    assert validation_metrics is not None

    total_return = cumulative_returns[-1] - 1
    max_drawdown = abs(cold_start_metrics["Max Drawdown"])
    expected_cold_calmar = (
        (1 + total_return) ** (TRADING_DAYS_PER_YEAR / 3) - 1
    ) / max_drawdown
    expected_validation_calmar = (
        (1 + total_return) ** (TRADING_DAYS_PER_YEAR / 4) - 1
    ) / max_drawdown

    assert cold_start_metrics["Calmar Ratio"] == pytest.approx(expected_cold_calmar)
    assert validation_metrics["Calmar Ratio"] == pytest.approx(
        expected_validation_calmar
    )
    assert validation_metrics["Calmar Ratio"] != pytest.approx(
        cold_start_metrics["Calmar Ratio"]
    )


def test_drawdown_includes_initial_capital_baseline() -> None:
    """Verify that entry costs create drawdown from initial capital."""
    dates = [datetime.date(2020, 1, day) for day in range(1, 4)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0] * 3})
    backtester = Backtester(
        data,
        BuyAndHoldStrategy({}),
        initial_capital=100_000.0,
        transaction_cost_bps=500.0,
        slippage_bps=0.0,
    )

    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    assert metrics["Total Return"] == pytest.approx(-0.05)
    assert metrics["Max Drawdown"] == pytest.approx(-0.05)
    assert metrics["Max Drawdown Duration"] == 3


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
            strict=True,
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


def test_pairs_returns_are_invariant_to_second_leg_price_scale() -> None:
    """Verify that quotation scale does not change pair returns or costs."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(80)]
    close_2 = [100.0 + i for i in range(80)]
    residuals = [0.0] * 80
    for index in range(40, 50):
        residuals[index] = 20.0
    for index in range(60, 70):
        residuals[index] = -20.0
    close_1 = [10.0 + 2.0 * close_2[index] + residuals[index] for index in range(80)]
    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})
    scaled_data = data.with_columns((pl.col("Close_2") * 10.0).alias("Close_2"))
    params = {"window": 10, "entry_z_score": 1.5, "exit_z_score": 0.5}

    base_results = Backtester(data, PairsTradingStrategy(params)).run()
    scaled_results = Backtester(
        scaled_data,
        PairsTradingStrategy(params),
    ).run()

    assert base_results["signal"].to_list() == scaled_results["signal"].to_list()
    for column in [
        "leg_1_weight",
        "leg_2_weight",
        "trade_turnover",
        "gross_strategy_returns",
        "transaction_costs",
        "strategy_returns",
        "cumulative_returns",
    ]:
        assert base_results[column].to_list() == pytest.approx(
            scaled_results[column].to_list()
        )


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
                datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
                datetime.datetime(2020, 1, 2, tzinfo=datetime.UTC),
                datetime.datetime(2020, 1, 3, tzinfo=datetime.UTC),
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


def test_save_results_persists_distinct_trading_assumptions(
    mock_polars_data: pl.DataFrame,
    sqlite_store,
) -> None:
    """Verify runs differing only in trading assumptions are both saved."""
    strategy = MovingAverageCrossoverStrategy({"short_window": 3, "long_window": 8})
    memory_store = InMemoryStrategyStore()

    default_bt = Backtester(mock_polars_data, strategy, tickers="AAPL")
    default_bt.run()
    custom_bt = Backtester(
        mock_polars_data,
        strategy,
        initial_capital=250_000.0,
        transaction_cost_bps=15.0,
        slippage_bps=7.5,
        tickers="AAPL",
    )
    custom_bt.run()

    for store in (sqlite_store, memory_store):
        assert default_bt.save_results(store=store) is True
        assert custom_bt.save_results(store=store) is True
        # Duplicate custom run should still deduplicate cleanly.
        assert custom_bt.save_results(store=store) is False

        records = store.list_recent()
        assert len(records) == 2
        saved_params = [record.parameters for record in records]
        assert {"short_window": 3, "long_window": 8} in saved_params
        assert {
            "short_window": 3,
            "long_window": 8,
            "initial_capital": 250_000.0,
            "transaction_cost_bps": 15.0,
            "slippage_bps": 7.5,
        } in saved_params


def test_get_performance_metrics_excludes_cold_start_row_from_risk_ratios() -> None:
    """Verify cold-start row 0 is excluded from Sharpe and Sortino only."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
    data = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 110.0, 99.0, 108.9, 103.455],
        }
    )
    backtester = Backtester(
        data,
        BuyAndHoldStrategy({}),
        initial_capital=100_000.0,
        transaction_cost_bps=50.0,
        slippage_bps=0.0,
    )
    results = backtester.run()

    cold_metrics = backtester.get_performance_metrics(include_initial_period=False)
    validation_metrics = backtester.get_performance_metrics(include_initial_period=True)
    assert cold_metrics is not None
    assert validation_metrics is not None

    all_returns = [float(val) for val in results["strategy_returns"].to_list()]
    active_returns = all_returns[1:]

    def _expected_sharpe(values: list[float]) -> float:
        mean_val = sum(values) / len(values)
        var_val = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        return (TRADING_DAYS_PER_YEAR**0.5) * mean_val / math.sqrt(var_val)

    def _expected_sortino(values: list[float]) -> float:
        mean_val = sum(values) / len(values)
        downside = math.sqrt(sum(min(v, 0.0) ** 2 for v in values) / len(values))
        return (TRADING_DAYS_PER_YEAR**0.5) * mean_val / downside

    assert cold_metrics["Sharpe Ratio"] == pytest.approx(
        _expected_sharpe(active_returns)
    )
    assert cold_metrics["Sortino Ratio"] == pytest.approx(
        _expected_sortino(active_returns)
    )
    assert validation_metrics["Sharpe Ratio"] == pytest.approx(
        _expected_sharpe(all_returns)
    )
    assert validation_metrics["Sortino Ratio"] == pytest.approx(
        _expected_sortino(all_returns)
    )
    assert cold_metrics["Sharpe Ratio"] != pytest.approx(
        validation_metrics["Sharpe Ratio"]
    )
    assert cold_metrics["Sortino Ratio"] != pytest.approx(
        validation_metrics["Sortino Ratio"]
    )

    # Row-0 entry cost and turnover still count in return, cost, and drawdown.
    for key in (
        "Total Return",
        "Gross Total Return",
        "Total Costs",
        "Trade Events",
        "Total Turnover",
        "Max Drawdown",
    ):
        assert cold_metrics[key] == pytest.approx(validation_metrics[key])
    assert cold_metrics["Total Costs"] == pytest.approx(0.005)
    assert cold_metrics["Trade Events"] == 1.0
    assert cold_metrics["Total Turnover"] == 1.0

    # A single cold-start row produces NaN risk ratios when row 0 is dropped.
    single_row_bt = Backtester(
        data.head(1),
        BuyAndHoldStrategy({}),
        transaction_cost_bps=50.0,
        slippage_bps=0.0,
    )
    single_row_bt.run()
    single_metrics = single_row_bt.get_performance_metrics(include_initial_period=False)
    assert single_metrics is not None
    assert math.isnan(single_metrics["Sharpe Ratio"])
    assert math.isnan(single_metrics["Sortino Ratio"])


def test_backtester_clamps_bankrupt_equity_at_zero() -> None:
    """Verify >100% single-day loss absorbs cumulative equity at zero."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
    # Short entry on day 1; day 2 jumps +150% (-150% short loss), followed by
    # -20% and +50% moves that must not revive bankrupt equity.
    data = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 100.0, 250.0, 200.0, 300.0],
        }
    )
    strategy = MockHoldingStrategy({"signals": [0.0, -1.0, -1.0, -1.0, -1.0]})
    backtester = Backtester(
        data,
        strategy,
        initial_capital=100_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    assert results["gross_cumulative_returns"].to_list() == pytest.approx(
        [1.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert results["cumulative_returns"].to_list() == pytest.approx(
        [1.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert results["equity_curve"].to_list() == pytest.approx(
        [100_000.0, 100_000.0, 0.0, 0.0, 0.0]
    )
    assert (results["equity_curve"] >= 0.0).all()
    assert metrics["Total Return"] == pytest.approx(-1.0)
    assert metrics["Gross Total Return"] == pytest.approx(-1.0)
    assert metrics["Max Drawdown"] == pytest.approx(-1.0)
    assert math.isnan(metrics["Calmar Ratio"])


def test_bankrupt_strategy_cannot_score_positive_sharpe_or_sortino() -> None:
    """Verify post-bankruptcy bars do not yield positive Sharpe or Sortino."""
    n_post_days = 20
    # Day 0: flat at 100; day 1: short at 100; day 2: jumps to 250 (-150%
    # wipeout); days 3..22: price drops 10% per day (+10% raw short returns).
    closes = [100.0, 100.0, 250.0]
    for _ in range(n_post_days):
        closes.append(closes[-1] * 0.9)
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
        for i in range(len(closes))
    ]
    data = pl.DataFrame({"Date": dates, "Close": closes})
    signals = [0.0] + [-1.0] * (len(closes) - 1)
    backtester = Backtester(
        data,
        MockHoldingStrategy({"signals": signals}),
        initial_capital=100_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    assert metrics["Total Return"] == pytest.approx(-1.0)
    assert metrics["Sharpe Ratio"] < 0.0
    assert metrics["Sortino Ratio"] < 0.0
