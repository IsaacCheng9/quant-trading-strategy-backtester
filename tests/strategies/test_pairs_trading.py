"""
Tests for the Pairs Trading strategy class.
"""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from quant_trading_strategy_backtester.backtester import (
    Backtester,
    build_trade_ledger,
)
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)


def _make_shocked_pair_data(num_days: int = 100) -> pl.DataFrame:
    """Create a pair with a stable beta and temporary residual shocks."""
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    close_2 = [100.0 + i for i in range(num_days)]
    residuals = [0.0] * num_days
    for i in range(50, 60):
        residuals[i] = 20.0
    for i in range(70, 80):
        residuals[i] = -20.0

    close_1 = [10.0 + 2.0 * close_2[i] + residuals[i] for i in range(num_days)]
    return pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})


def test_pairs_trading_strategy_generate_signals() -> None:
    # Create mock data for two assets
    start_date = date(2020, 1, 1)
    end_date = date(2020, 4, 9)
    num_days = (end_date - start_date).days + 1

    date_range = [start_date + timedelta(days=i) for i in range(num_days)]

    data = pl.DataFrame(
        {
            "Date": pl.Series(date_range),
            "Close_1": pl.arange(0, num_days, eager=True).cum_sum() + 100,
            "Close_2": pl.arange(0, num_days, eager=True).cum_sum() * 0.5 + 100,
        }
    )

    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {
        "hedge_ratio",
        "spread",
        "z_score",
        "signal",
        "leg_1_weight",
        "leg_2_weight",
        "leg_1_weight_change",
        "leg_2_weight_change",
        "position_change",
    }
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0, -1.0]).all()


def test_pairs_trading_strategy_initialisation() -> None:
    params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    assert strategy.window == 20
    assert strategy.entry_z_score == 2.0
    assert strategy.exit_z_score == 0.5


def test_pairs_trading_strategy_rejects_invalid_thresholds() -> None:
    params = {"window": 20, "entry_z_score": 1.0, "exit_z_score": 1.0}

    with pytest.raises(ValueError, match="exit_z_score must be less"):
        PairsTradingStrategy(params)


def test_pairs_trading_strategy_signal_generation() -> None:
    data = _make_shocked_pair_data()
    params = {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    assert signals.filter(pl.col("signal") == 1.0).height > 0
    assert signals.filter(pl.col("signal") == -1.0).height > 0
    assert signals.filter(pl.col("signal") == 0.0).height > 0


def test_pairs_trading_strategy_with_invalid_data() -> None:
    data = pl.DataFrame(
        {
            "Close_1": [100, 101, 102],
            "Close_3": [100, 101, 102],  # Invalid column name
        }
    )
    params = {"window": 2, "entry_z_score": 2.0, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    with pytest.raises(
        ValueError, match="Data must contain 'Close_1' and 'Close_2' columns"
    ):
        strategy.generate_signals(data)


def test_pairs_trading_strategy_with_mock_polars_data():
    """Test the Pairs Trading strategy with mock data."""
    mock_polars_data = _make_shocked_pair_data()

    # Strategy parameters
    params = {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.5}

    strategy = PairsTradingStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(mock_polars_data)

    # Check if signals are generated correctly
    assert signals["signal"].null_count() == 0, "Null signals generated"

    long_signals = signals.filter(pl.col("signal") == 1)
    short_signals = signals.filter(pl.col("signal") == -1)
    exit_signals = signals.filter(pl.col("signal") == 0)
    assert len(long_signals) > 0, "No long signals (1) generated"
    assert len(short_signals) > 0, "No short signals (-1) generated"
    assert len(exit_signals) > 0, "No exit signals (0) generated"

    # Check if the strategy generates a signal when z-score crosses thresholds
    entry_long = signals.filter(
        (pl.col("z_score") < -params["entry_z_score"]) & (pl.col("signal") == 1)
    )
    entry_short = signals.filter(
        (pl.col("z_score") > params["entry_z_score"]) & (pl.col("signal") == -1)
    )
    exit_positions = signals.filter(
        (pl.col("z_score").abs() < params["exit_z_score"]) & (pl.col("signal") == 0)
    )
    assert len(entry_long) > 0, "No long entry signals generated"
    assert len(entry_short) > 0, "No short entry signals generated"
    assert len(exit_positions) > 0, "No exit signals generated"

    # Check if position changes are calculated correctly
    non_zero_changes = signals.filter(pl.col("position_change") != 0)
    assert len(non_zero_changes) > 0, "No position changes"
    assert signals["position_change"].abs().sum() > 0, "No position changes"

    # Check if the spread and z-score are calculated correctly.
    valid_spreads = signals.filter(pl.col("hedge_ratio").is_not_null())
    spread_error = (
        valid_spreads["spread"]
        - valid_spreads["Close_1"]
        + valid_spreads["hedge_ratio"] * valid_spreads["Close_2"]
    )
    max_spread_error = max(abs(float(error)) for error in spread_error.to_list())
    assert max_spread_error < 1e-10, "Spread calculation is incorrect"
    assert signals["z_score"].null_count() == 0, "Z-score contains null values"


def test_pairs_trading_strategy_estimates_hedge_ratio() -> None:
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(40)]
    close_2 = [100.0 + i for i in range(40)]
    close_1 = [15.0 + 2.0 * price for price in close_2]
    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})

    strategy = PairsTradingStrategy(
        {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5}
    )
    signals = strategy.generate_signals(data)
    valid_signals = signals.filter(pl.col("hedge_ratio").is_not_null())

    assert valid_signals["hedge_ratio"].to_list() == pytest.approx([2.0] * 31)
    assert valid_signals["spread"].to_list() == pytest.approx([15.0] * 31)


def test_pairs_trading_strategy_is_invariant_to_second_leg_price_scale() -> None:
    data = _make_shocked_pair_data(num_days=80)
    scaled_data = data.with_columns((pl.col("Close_2") * 10.0).alias("Close_2"))
    params = {"window": 10, "entry_z_score": 1.5, "exit_z_score": 0.5}

    base_signals = PairsTradingStrategy(params).generate_signals(data)
    scaled_signals = PairsTradingStrategy(params).generate_signals(scaled_data)

    assert base_signals["spread"].to_list() == pytest.approx(
        scaled_signals["spread"].to_list(), nan_ok=True
    )
    assert base_signals["spread_mean"].to_list() == pytest.approx(
        scaled_signals["spread_mean"].to_list(), nan_ok=True
    )
    assert base_signals["spread_std"].to_list() == pytest.approx(
        scaled_signals["spread_std"].to_list(), nan_ok=True
    )
    assert base_signals["z_score"].to_list() == pytest.approx(
        scaled_signals["z_score"].to_list(), nan_ok=True
    )


def _make_indicator_frame(
    z_scores: list[float],
    close_1: list[float] | None = None,
    close_2: list[float] | None = None,
    hedge_ratios: list[float | None] | None = None,
    spread_stds: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a synthetic indicator DataFrame for stateful signal tests."""
    num_rows = len(z_scores)
    return pl.DataFrame(
        {
            "Close_1": close_1 if close_1 is not None else [100.0] * num_rows,
            "Close_2": close_2 if close_2 is not None else [50.0] * num_rows,
            "hedge_ratio": (
                hedge_ratios if hedge_ratios is not None else [2.0] * num_rows
            ),
            "spread": [0.0] * num_rows,
            "spread_mean": [0.0] * num_rows,
            "spread_std": (
                spread_stds if spread_stds is not None else [1.0] * num_rows
            ),
            "z_score": z_scores,
        }
    )


def _make_gap_pair_data(
    residual_overrides: dict[int, float], num_days: int = 40
) -> pl.DataFrame:
    """Create a pair with stable positive beta and controlled shock days."""
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    # Alternate Close_2 widely around 100 and pin shock days to 100 so residual
    # shocks perturb the rolling OLS beta minimally.
    close_2 = [50.0 if i % 2 == 0 else 150.0 for i in range(num_days)]
    for day in residual_overrides:
        close_2[day] = 100.0
    residuals = [0.2 if i % 2 == 0 else -0.2 for i in range(num_days)]
    for day, residual in residual_overrides.items():
        residuals[day] = residual
    close_1 = [10.0 + 2.0 * close_2[i] + residuals[i] for i in range(num_days)]
    return pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})


def test_pairs_trading_long_exits_on_gap_through_exit_band() -> None:
    """Verify a long position exits when z-score gaps past +exit_z_score."""
    data = _make_gap_pair_data({30: -4.0, 31: -2.0, 32: 1.5})

    params = {"window": 15, "entry_z_score": 2.0, "exit_z_score": 0.5}
    signals = PairsTradingStrategy(params).generate_signals(data)

    assert signals["z_score"][30] < -2.0
    assert signals["signal"][30] == 1.0
    assert -2.0 < signals["z_score"][31] < -0.5
    assert signals["signal"][31] == 1.0
    # Confirm day 32 genuinely gapped past +exit_z_score while below +entry.
    assert 0.5 < signals["z_score"][32] < 2.0
    assert signals["signal"][32] == 0.0
    assert signals["leg_1_weight"][32] == 0.0
    assert signals["leg_2_weight"][32] == 0.0
    assert signals["position_change"][32] == -1.0


def test_pairs_trading_short_exits_on_gap_through_exit_band() -> None:
    """Verify a short position exits when z-score gaps past -exit_z_score."""
    data = _make_gap_pair_data({30: 4.0, 31: 2.0, 32: -1.5})

    params = {"window": 15, "entry_z_score": 2.0, "exit_z_score": 0.5}
    signals = PairsTradingStrategy(params).generate_signals(data)

    assert signals["z_score"][30] > 2.0
    assert signals["signal"][30] == -1.0
    assert 0.5 < signals["z_score"][31] < 2.0
    assert signals["signal"][31] == -1.0
    # Confirm day 32 genuinely gapped past -exit_z_score while above -entry.
    assert -2.0 < signals["z_score"][32] < -0.5
    assert signals["signal"][32] == 0.0
    assert signals["leg_1_weight"][32] == 0.0
    assert signals["leg_2_weight"][32] == 0.0
    assert signals["position_change"][32] == 1.0


def test_pairs_trading_direct_flip_on_opposite_entry_gap() -> None:
    """Verify a position flips directly on an opposite-entry z-score gap."""
    data = _make_gap_pair_data({30: -4.0, 31: 12.0})

    params = {"window": 15, "entry_z_score": 2.0, "exit_z_score": 0.5}
    signals = PairsTradingStrategy(params).generate_signals(data)

    assert signals["z_score"][30] < -2.0
    assert signals["signal"][30] == 1.0
    assert signals["leg_1_weight"][30] > 0.0
    assert signals["leg_2_weight"][30] < 0.0

    assert signals["z_score"][31] > 2.0
    assert signals["signal"][31] == -1.0
    assert signals["position_change"][31] == -2.0
    assert signals["leg_1_weight"][31] < 0.0
    assert signals["leg_2_weight"][31] > 0.0
    assert abs(signals["leg_1_weight"][31]) + abs(
        signals["leg_2_weight"][31]
    ) == pytest.approx(1.0)


def test_pairs_trading_blocks_negative_hedge_ratio_entries() -> None:
    """Verify negative rolling beta blocks entries and same-sign legs."""
    start_date = date(2020, 1, 1)
    num_days = 80
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    close_2 = [100.0 + i for i in range(num_days)]
    residuals = [0.0] * num_days
    for i in range(40, 50):
        residuals[i] = 5.0
    for i in range(60, 70):
        residuals[i] = -5.0

    # Negative slope keeps rolling OLS hedge_ratio < 0 across the series.
    close_1 = [800.0 - 3.0 * close_2[i] + residuals[i] for i in range(num_days)]
    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})

    params = {"window": 15, "entry_z_score": 1.5, "exit_z_score": 0.5}
    signals = PairsTradingStrategy(params).generate_signals(data)

    valid_betas = signals.filter(pl.col("hedge_ratio").is_not_null())["hedge_ratio"]
    assert (valid_betas < 0.0).all()
    assert (signals["signal"] == 0.0).all()
    assert (signals["leg_1_weight"] == 0.0).all()
    assert (signals["leg_2_weight"] == 0.0).all()
    assert (signals["leg_1_weight"] * signals["leg_2_weight"] <= 0.0).all()


def test_pairs_trading_closes_open_position_when_hedge_ratio_turns_non_positive() -> (
    None
):
    """Verify an open trade flattens if rolling beta turns non-positive."""
    strategy = PairsTradingStrategy(
        {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5}
    )
    indicators = _make_indicator_frame(
        z_scores=[0.0, -2.5, -1.5, -1.5, -2.5, -1.0],
        hedge_ratios=[2.0, 2.0, 1.8, -0.5, 0.0, 2.0],
    )

    signal_values, leg_1_weights, leg_2_weights = strategy._generate_stateful_signals(
        indicators
    )

    assert signal_values == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    # Entry weight is locked on hold day (index 2), then flattened on index 3.
    assert leg_1_weights[1] > 0.0
    assert leg_2_weights[1] < 0.0
    assert leg_1_weights[2] == pytest.approx(leg_1_weights[1])
    assert leg_2_weights[2] == pytest.approx(leg_2_weights[1])
    assert leg_1_weights[3:] == [0.0, 0.0, 0.0]
    assert leg_2_weights[3:] == [0.0, 0.0, 0.0]


def test_pairs_trading_locks_leg_weights_while_holding_and_limits_trade_events() -> (
    None
):
    """Verify held trades keep constant weights without inflating events."""
    data = _make_shocked_pair_data(num_days=100)
    params = {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.5}
    strategy = PairsTradingStrategy(params)
    signals = strategy.generate_signals(data)

    # Confirm there are multiple consecutive holding days in the test series.
    hold_rows = signals.with_columns(
        pl.col("signal").shift(1).fill_null(0.0).alias("prev_signal")
    ).filter((pl.col("signal") != 0.0) & (pl.col("signal") == pl.col("prev_signal")))
    assert hold_rows.height > 5
    assert (hold_rows["leg_1_weight_change"] == 0.0).all()
    assert (hold_rows["leg_2_weight_change"] == 0.0).all()

    # Check weights stay strictly equal to the previous row on every hold day.
    weights_with_prev = signals.with_columns(
        [
            pl.col("signal").shift(1).fill_null(0.0).alias("prev_signal"),
            pl.col("leg_1_weight").shift(1).fill_null(0.0).alias("prev_w1"),
            pl.col("leg_2_weight").shift(1).fill_null(0.0).alias("prev_w2"),
        ]
    ).filter((pl.col("signal") != 0.0) & (pl.col("signal") == pl.col("prev_signal")))
    assert weights_with_prev["leg_1_weight"].to_list() == pytest.approx(
        weights_with_prev["prev_w1"].to_list()
    )
    assert weights_with_prev["leg_2_weight"].to_list() == pytest.approx(
        weights_with_prev["prev_w2"].to_list()
    )

    # Run Backtester and confirm Trade Events matches signal transitions.
    backtester = Backtester(data, strategy)
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    assert metrics is not None

    signal_transitions = results.filter(pl.col("position_change") != 0.0).height
    assert signal_transitions > 0
    assert metrics["Trade Events"] == float(signal_transitions)

    ledger = build_trade_ledger(results)
    assert ledger.height == signal_transitions
    assert "Rebalance" not in ledger["Action"].to_list()


def test_pairs_trading_rejects_non_positive_prices_and_flattens_open_trades() -> None:
    """Verify zero or negative prices block entry and flatten open trades."""
    strategy = PairsTradingStrategy(
        {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5}
    )
    indicators = pl.DataFrame(
        {
            "Close_1": [100.0, 0.0, 100.0, 100.0, 100.0, 0.0, 100.0, -5.0],
            "Close_2": [50.0, 50.0, -50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            "hedge_ratio": [2.0] * 8,
            "spread": [10.0] * 8,
            "spread_mean": [10.0] * 8,
            "spread_std": [1.0] * 8,
            "z_score": [0.0, -2.5, -2.5, -2.5, -1.5, -1.5, -2.5, -1.5],
        }
    )

    signal_values, leg_1_weights, leg_2_weights = strategy._generate_stateful_signals(
        indicators
    )

    # Indices 1 (Close_1=0) and 2 (Close_2=-50) must stay flat; index 3 enters
    # long, index 5 flattens on Close_1=0, index 6 re-enters, index 7 flattens
    # on Close_1=-5.
    assert signal_values == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert leg_1_weights[1:3] == [0.0, 0.0]
    assert leg_2_weights[1:3] == [0.0, 0.0]
    assert leg_1_weights[3] > 0.0
    assert leg_2_weights[3] < 0.0
    assert abs(leg_1_weights[3]) + abs(leg_2_weights[3]) == pytest.approx(1.0)
    assert leg_1_weights[5] == 0.0
    assert leg_2_weights[5] == 0.0
    assert leg_1_weights[7] == 0.0
    assert leg_2_weights[7] == 0.0

    # End-to-end check that non-positive Close rows produce z_score == 0.0.
    data = _make_shocked_pair_data(num_days=85)
    bad_prices = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 52)
        .then(0.0)
        .when(pl.int_range(pl.len()) == 54)
        .then(-10.0)
        .otherwise(pl.col("Close_2"))
        .alias("Close_2")
    )
    signals = PairsTradingStrategy(
        {"window": 10, "entry_z_score": 1.5, "exit_z_score": 0.5}
    ).generate_signals(bad_prices)
    assert signals["z_score"][52] == 0.0
    assert signals["signal"][52] == 0.0
    assert signals["z_score"][54] == 0.0
    assert signals["signal"][54] == 0.0


def test_pairs_trading_spread_stats_match_window_consistent_reference() -> None:
    """Verify spread stats use the current window beta across the lookback."""
    window = 10
    num_days = 50
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Vary Close_2 and shift the true slope across regimes so rolling beta_t
    # changes materially over time.
    close_2 = [100.0 + 0.8 * i + 6.0 * math.sin(i * 0.5) for i in range(num_days)]
    close_1 = []
    for i, c2 in enumerate(close_2):
        slope = 1.0 + 2.5 * (i / (num_days - 1))
        residual = 3.0 * math.cos(i * 0.9) + (1.5 if i % 5 == 0 else -0.5)
        close_1.append(20.0 + slope * c2 + residual)

    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})
    strategy = PairsTradingStrategy(
        {"window": window, "entry_z_score": 1.5, "exit_z_score": 0.5}
    )
    signals = strategy.generate_signals(data)

    # Output schema must stay unchanged without leaking intermediate columns.
    expected_columns = [
        "Date",
        "Close_1",
        "Close_2",
        "hedge_ratio",
        "spread",
        "spread_mean",
        "spread_std",
        "z_score",
        "signal",
        "leg_1_weight",
        "leg_2_weight",
        "leg_1_weight_change",
        "leg_2_weight_change",
        "position_change",
    ]
    assert signals.columns == expected_columns

    # Warmup rows before index (window - 1) have null spread stats and 0 z.
    for idx in range(window - 1):
        assert signals["hedge_ratio"][idx] is None
        assert signals["spread"][idx] is None
        assert signals["spread_mean"][idx] is None
        assert signals["spread_std"][idx] is None
        assert signals["z_score"][idx] == 0.0

    # First valid row must be at index (window - 1), not (2 * window - 2).
    assert signals["hedge_ratio"][window - 1] is not None
    assert signals["spread_mean"][window - 1] is not None
    assert signals["spread_std"][window - 1] is not None
    assert signals["spread_std"][window - 1] > 0.0

    ref_betas: list[float] = []
    for t in range(window - 1, num_days):
        c1_win = close_1[t - window + 1 : t + 1]
        c2_win = close_2[t - window + 1 : t + 1]
        mean_1 = sum(c1_win) / window
        mean_2 = sum(c2_win) / window
        var_2 = sum((y - mean_2) ** 2 for y in c2_win) / (window - 1)
        cov_12 = sum(
            (x - mean_1) * (y - mean_2) for x, y in zip(c1_win, c2_win, strict=True)
        ) / (window - 1)
        beta_t = cov_12 / var_2
        ref_betas.append(beta_t)

        window_spread = [x - beta_t * y for x, y in zip(c1_win, c2_win, strict=True)]
        ref_mean = sum(window_spread) / window
        ref_var = sum((s - ref_mean) ** 2 for s in window_spread) / (window - 1)
        ref_std = math.sqrt(ref_var)
        ref_z = (
            (window_spread[-1] - ref_mean) / ref_std
            if beta_t > 0.0 and ref_std > 0.0
            else 0.0
        )

        assert signals["hedge_ratio"][t] == pytest.approx(beta_t, rel=1e-10)
        assert signals["spread"][t] == pytest.approx(window_spread[-1], rel=1e-10)
        assert signals["spread_mean"][t] == pytest.approx(ref_mean, rel=1e-10)
        assert signals["spread_std"][t] == pytest.approx(ref_std, rel=1e-10)
        assert signals["z_score"][t] == pytest.approx(ref_z, rel=1e-10)

    # Confirm rolling beta changed materially and the naive rolling-spread
    # method over mixed historical betas would diverge visibly.
    assert max(ref_betas) - min(ref_betas) > 1.0
    naive_stats = signals.select(
        naive_mean=pl.col("spread").rolling_mean(
            window_size=window, min_samples=window
        ),
        naive_std=pl.col("spread").rolling_std(window_size=window, min_samples=window),
    )
    mean_diffs = [
        abs(signals["spread_mean"][t] - naive_stats["naive_mean"][t])
        for t in range(2 * window - 2, num_days)
    ]
    std_diffs = [
        abs(signals["spread_std"][t] - naive_stats["naive_std"][t])
        for t in range(2 * window - 2, num_days)
    ]
    assert max(mean_diffs) > 5.0
    assert max(std_diffs) > 1.0


@pytest.mark.parametrize(
    ("scale", "step"),
    [
        (1e-4, 1.0),
        (1.0, 1.0),
        (1.0, 0.3),
        (1e6, 1.0),
    ],
)
def test_pairs_trading_collinear_series_sets_zero_std_and_z_score(
    scale: float, step: float
) -> None:
    """Verify exact collinearity yields zero spread_std and zero z_score."""
    window = 10
    num_days = 40
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    close_2 = [scale * (100.0 + step * i) for i in range(num_days)]
    intercept = 15.0 * scale
    close_1 = [intercept + 2.0 * price for price in close_2]
    data = pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})

    strategy = PairsTradingStrategy(
        {"window": window, "entry_z_score": 2.0, "exit_z_score": 0.5}
    )
    signals = strategy.generate_signals(data)

    for idx in range(window - 1):
        assert signals["spread_std"][idx] is None
        assert signals["z_score"][idx] == 0.0

    valid = signals.slice(window - 1)
    assert valid["hedge_ratio"].to_list() == pytest.approx(
        [2.0] * (num_days - window + 1)
    )
    assert valid["spread_mean"].to_list() == pytest.approx(
        [intercept] * (num_days - window + 1)
    )
    assert (valid["spread_std"] == 0.0).all()
    assert (valid["z_score"] == 0.0).all()
    assert (valid["signal"] == 0.0).all()
