"""Tests for Streamlit app helpers."""

import datetime
from typing import Any

import polars as pl
import pytest

from quant_trading_strategy_backtester import app, data, optimiser, visualisation
from quant_trading_strategy_backtester.app import (
    _build_save_key,
    _display_benchmark_comparison,
    _format_historical_sharpe,
    _get_benchmark_date_range,
    _get_final_backtest_data,
    _get_prior_date,
    _get_strategy_store,
    _streamlit_progress,
    prepare_pairs_trading_strategy_without_optimisation,
    prepare_single_ticker_strategy,
)
from quant_trading_strategy_backtester.optimiser import (
    WALK_FORWARD_FOLDS,
    stitch_walk_forward_oos_results,
    walk_forward_optimise,
)
from quant_trading_strategy_backtester.storage import (
    InMemoryStrategyStore,
    SQLiteStrategyStore,
)
from quant_trading_strategy_backtester.streamlit_ui import BacktestConfig


def test_get_benchmark_date_range_returns_inclusive_dates() -> None:
    """Verify benchmark loading requests inclusive start and end dates."""
    data = pl.DataFrame(
        {
            "Date": [
                datetime.datetime(2020, 1, 2, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2020, 1, 3, 0, 0, tzinfo=datetime.UTC),
            ],
            "Close": [100.0, 101.0],
        }
    )

    start_date, end_date = _get_benchmark_date_range(data)

    assert start_date == datetime.date(2020, 1, 2)
    assert end_date == datetime.date(2020, 1, 3)


def test_build_save_key_changes_with_dates_modes_and_assumptions() -> None:
    """Verify save keys distinguish dates, modes, assumptions, and params."""
    base_kwargs = {
        "strategy_type": "Moving Average Crossover",
        "strategy_params": {"short_window": 5, "long_window": 20},
        "tickers": "AAPL",
        "start_date": datetime.date(2020, 1, 1),
        "end_date": datetime.date(2020, 12, 31),
        "optimise": False,
        "walk_forward": False,
        "auto_select_tickers": False,
        "initial_capital": 100_000.0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 3.0,
    }
    base_key = _build_save_key(**base_kwargs)

    # Reordered dict keys should map to the same canonical key.
    reordered_key = _build_save_key(
        **{
            **base_kwargs,
            "strategy_params": {"long_window": 20, "short_window": 5},
        }
    )
    assert reordered_key == base_key

    assert (
        _build_save_key(**{**base_kwargs, "start_date": datetime.date(2021, 1, 1)})
        != base_key
    )
    assert (
        _build_save_key(**{**base_kwargs, "end_date": datetime.date(2021, 12, 31)})
        != base_key
    )
    assert _build_save_key(**{**base_kwargs, "optimise": True}) != base_key
    assert _build_save_key(**{**base_kwargs, "walk_forward": True}) != base_key
    assert _build_save_key(**{**base_kwargs, "auto_select_tickers": True}) != base_key
    assert _build_save_key(**{**base_kwargs, "initial_capital": 250_000.0}) != base_key
    assert _build_save_key(**{**base_kwargs, "transaction_cost_bps": 10.0}) != base_key
    assert _build_save_key(**{**base_kwargs, "slippage_bps": 8.0}) != base_key


def test_format_historical_sharpe() -> None:
    """Verify Sharpe formatting handles zero, finite, NaN, inf, and None."""
    assert _format_historical_sharpe(0.0) == "0.00"
    assert _format_historical_sharpe(1.234) == "1.23"
    assert _format_historical_sharpe(-0.5) == "-0.50"
    assert _format_historical_sharpe(float("nan")) == "N/A"
    assert _format_historical_sharpe(float("inf")) == "N/A"
    assert _format_historical_sharpe(float("-inf")) == "N/A"
    assert _format_historical_sharpe(None) == "N/A"


def test_get_strategy_store_selects_in_memory_on_streamlit_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Streamlit Cloud environment selects InMemoryStrategyStore."""
    monkeypatch.setenv("STREAMLIT_SHARING_MODE", "streamlit")
    monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(app.st, "session_state", fake_session_state)

    store = _get_strategy_store()

    assert isinstance(store, InMemoryStrategyStore)
    assert "strategy_results" in fake_session_state


def test_get_strategy_store_selects_sqlite_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify local writable environment selects SQLiteStrategyStore."""
    monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)
    monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)

    store = _get_strategy_store()

    assert isinstance(store, SQLiteStrategyStore)


def test_get_strategy_store_falls_back_to_in_memory_when_cwd_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify read-only working directory falls back to session-state store."""
    monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)
    monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)
    monkeypatch.setattr(app.os, "access", lambda *_args, **_kwargs: False)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(app.st, "session_state", fake_session_state)

    store = _get_strategy_store()

    assert isinstance(store, InMemoryStrategyStore)
    assert "strategy_results" in fake_session_state


def test_get_benchmark_date_range_starts_from_prior_date() -> None:
    """Verify benchmark loading starts on prior_date for validation slices."""
    validation_data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 6),
                datetime.date(2020, 1, 7),
            ],
            "Close": [101.0, 102.0],
        }
    )

    start_date, end_date = _get_benchmark_date_range(
        validation_data,
        prior_date=datetime.date(2020, 1, 3),
    )

    assert start_date == datetime.date(2020, 1, 3)
    assert end_date == datetime.date(2020, 1, 7)


def test_get_prior_date_returns_previous_date_for_validation_slice() -> None:
    """Verify _get_prior_date finds the last date strictly before a slice."""
    dates = [
        datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2020, 1, 2, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2020, 1, 3, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2020, 1, 6, 0, 0, tzinfo=datetime.UTC),
    ]
    full_data = pl.DataFrame({"Date": dates, "Close": [100.0, 101.0, 102.0, 103.0]})
    validation_slice = full_data[2:]

    assert _get_prior_date(full_data, validation_slice) == dates[1]
    assert _get_prior_date(full_data, full_data) is None


def test_display_benchmark_comparison_passes_prior_date_for_validation(
    monkeypatch,
) -> None:
    """Verify benchmark comparison fetches and aligns from prior_date."""
    dates = [
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
    ]
    full_data = pl.DataFrame({"Date": dates, "Close": [100.0, 102.0, 104.0, 106.0]})
    validation_data = full_data[1:]
    results = pl.DataFrame({"Date": dates[1:], "strategy_returns": [0.02, 0.01, 0.03]})

    loaded_range: list[tuple[datetime.date, datetime.date]] = []
    displayed_metrics: list[dict[str, float]] = []

    def mock_load_benchmark(
        ticker: str, start_date: datetime.date, end_date: datetime.date
    ) -> pl.DataFrame:
        loaded_range.append((start_date, end_date))
        return pl.DataFrame(
            {
                "Date": dates,
                "Close": [100.0, 104.0, 106.08, 108.2016],
            }
        )

    def mock_display_metrics(metrics: dict[str, float], ticker: str) -> None:
        displayed_metrics.append(metrics)

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        mock_load_benchmark,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.display_benchmark_metrics",
        mock_display_metrics,
    )

    _display_benchmark_comparison(results, validation_data, full_data=full_data)

    assert loaded_range == [(datetime.date(2020, 1, 1), datetime.date(2020, 1, 4))]
    assert len(displayed_metrics) == 1
    assert displayed_metrics[0]["Benchmark Observations"] == 3
    assert displayed_metrics[0]["Benchmark Total Return"] == pytest.approx(
        108.2016 / 100.0 - 1
    )


def test_streamlit_progress_updates_and_clears_widgets(monkeypatch) -> None:
    """Verify _streamlit_progress updates widgets and clears them on exit."""
    events: list[tuple[str, object]] = []

    class FakeProgressBar:
        def progress(self, value: float) -> None:
            events.append(("progress", value))

        def empty(self) -> None:
            events.append(("progress_empty", None))

    class FakeStatusText:
        def text(self, message: str) -> None:
            events.append(("text", message))

        def empty(self) -> None:
            events.append(("text_empty", None))

    monkeypatch.setattr(app.st, "progress", lambda _val: FakeProgressBar())
    monkeypatch.setattr(app.st, "empty", lambda: FakeStatusText())

    with (
        pytest.raises(RuntimeError, match="boom"),
        _streamlit_progress() as progress_cb,
    ):
        progress_cb(1, 2, "Halfway")
        raise RuntimeError("boom")

    assert events == [
        ("text", "Halfway"),
        ("progress", 0.5),
        ("progress_empty", None),
        ("text_empty", None),
    ]


def test_display_walk_forward_results_formats_non_finite_sharpes_as_na(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify display_walk_forward_results renders NaN and inf as N/A."""
    rendered_metrics: list[tuple[str, str]] = []

    class FakeExpander:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCol:
        def metric(self, label: str, value: str) -> None:
            rendered_metrics.append((label, value))

    monkeypatch.setattr(visualisation.st, "expander", lambda _label: FakeExpander())
    monkeypatch.setattr(visualisation.st, "subheader", lambda *_a, **_k: None)
    monkeypatch.setattr(visualisation.st, "write", lambda *_a, **_k: None)
    monkeypatch.setattr(
        visualisation.st,
        "columns",
        lambda _n: (FakeCol(), FakeCol()),
    )

    visualisation.display_walk_forward_results(
        [
            {
                "fold": 1,
                "train_rows": 20,
                "test_rows": 10,
                "params": {"window": 5},
                "in_sample_sharpe": 1.23456,
                "oos_metrics": {"Sharpe Ratio": float("nan")},
            },
            {
                "fold": 2,
                "train_rows": 30,
                "test_rows": 10,
                "params": {"window": 10},
                "in_sample_sharpe": float("inf"),
                "oos_metrics": {"Sharpe Ratio": -0.5},
            },
        ]
    )

    assert rendered_metrics == [
        ("In-Sample Sharpe", "1.2346"),
        ("Out-of-Sample Sharpe", "N/A"),
        ("In-Sample Sharpe", "N/A"),
        ("Out-of-Sample Sharpe", "-0.5000"),
    ]


def test_app_injects_cached_loaders_into_optimiser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify app preparation injects Streamlit cached loaders into optimiser."""
    captured: dict[str, object] = {}

    def fake_optimise_pairs(*args: object, **kwargs: object):
        captured["load_two_tickers"] = kwargs.get("load_two_tickers")
        captured["load_many_tickers"] = kwargs.get("load_many_tickers")
        captured["same_company_check"] = kwargs.get("same_company_check")
        return (
            ("AAPL", "MSFT"),
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            {},
        )

    monkeypatch.setattr(
        app,
        "_cached_get_top_sp500_companies",
        lambda _n: [("AAPL", 2.0), ("MSFT", 1.0)],
    )
    monkeypatch.setattr(app, "optimise_pairs_trading_tickers", fake_optimise_pairs)
    monkeypatch.setattr(
        app,
        "_cached_load_yfinance_data_two_tickers",
        lambda *_a, **_k: pl.DataFrame(
            {
                "Date": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
                "Close_1": [100.0, 101.0],
                "Close_2": [200.0, 202.0],
            }
        ),
    )

    app.prepare_pairs_trading_strategy_with_optimisation(
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 31),
        {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
        optimise=False,
    )

    assert captured["load_two_tickers"] is None
    assert captured["load_many_tickers"] is app._cached_load_yfinance_data_many_tickers
    assert captured["same_company_check"] is app._cached_is_same_company


def test_cached_load_yfinance_data_many_tickers_caches_batch_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _cached_load_yfinance_data_many_tickers caches repeated calls."""
    calls: list[tuple[str, ...]] = []
    expected = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
            "AAPL": [100.0, 101.0],
            "MSFT": [200.0, 202.0],
        }
    )

    def fake_load_many(tickers, start_date, end_date):
        calls.append(tuple(tickers))
        return expected

    monkeypatch.setattr(app, "load_yfinance_data_many_tickers", fake_load_many)

    first = app._cached_load_yfinance_data_many_tickers(
        ("AAPL", "MSFT"),
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 3),
    )
    second = app._cached_load_yfinance_data_many_tickers(
        ("AAPL", "MSFT"),
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 3),
    )

    assert calls == [("AAPL", "MSFT")]
    assert first.equals(expected)
    assert second.equals(expected)


def test_get_final_backtest_data_walk_forward_returns_stitched_oos_slice(
    monkeypatch,
) -> None:
    """Verify walk-forward final data covers all OOS folds and remainder."""
    info_messages: list[str] = []
    monkeypatch.setattr(app.st, "info", info_messages.append)

    n_rows = 63
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    data = pl.DataFrame({"Date": dates, "Close": [100.0 + i for i in range(n_rows)]})

    oos_data = _get_final_backtest_data(
        data, use_validation_data=True, walk_forward=True
    )

    # 63 // 6 = 10 initial training rows; 53 stitched OOS rows across 5 folds.
    assert len(oos_data) == 53
    assert oos_data["Date"][0] == dates[10]
    assert oos_data["Date"][-1] == dates[-1]
    assert len(info_messages) == 1
    assert "stitched walk-forward out-of-sample" in info_messages[0]
    assert "53 rows" in info_messages[0]
    assert f"{WALK_FORWARD_FOLDS} expanding folds" in info_messages[0]


def test_prepare_single_ticker_strategy_returns_stitched_walk_forward_oos(
    monkeypatch,
) -> None:
    """Verify walk-forward single-ticker prep retains stitched OOS output."""
    n_rows = 90
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    data = pl.DataFrame(
        {"Date": dates, "Close": [100.0 + i * 0.5 for i in range(n_rows)]}
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: data,
    )

    returned_data, ticker_display, best_params, walk_forward_oos = (
        prepare_single_ticker_strategy(
            "AAPL",
            dates[0],
            dates[-1],
            "Moving Average Crossover",
            {"short_window": [3, 5], "long_window": [10, 15]},
            optimise=True,
            walk_forward=True,
        )
    )

    assert len(returned_data) == n_rows
    assert ticker_display == "AAPL"
    assert set(best_params.keys()) == {"short_window", "long_window"}
    assert walk_forward_oos is not None
    stitched_results, stitched_metrics = walk_forward_oos
    # 90 // 6 = 15 initial training rows; 75 stitched OOS rows.
    assert len(stitched_results) == 75
    assert stitched_results["Date"][0] == dates[15]
    assert stitched_results["Date"][-1] == dates[-1]
    assert "Total Return" in stitched_metrics


def test_prepare_pairs_strategy_without_auto_select_returns_stitched_walk_forward_oos(
    monkeypatch,
) -> None:
    """Verify walk-forward manual pairs prep retains stitched OOS output."""
    n_rows = 72
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    pair_data = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": [100.0 + i * 0.2 for i in range(n_rows)],
            "Close_2": [200.0 + i * 0.4 for i in range(n_rows)],
        }
    )
    expected_results = pl.DataFrame(
        {
            "Date": dates[12:],
            "strategy_returns": [0.01] * 60,
            "gross_strategy_returns": [0.01] * 60,
            "transaction_costs": [0.0] * 60,
            "trade_turnover": [0.0] * 60,
            "cumulative_returns": [1.01] * 60,
            "gross_cumulative_returns": [1.01] * 60,
            "equity_curve": [101000.0] * 60,
            "cumulative_transaction_costs": [0.0] * 60,
        }
    )
    expected_metrics = {
        "Total Return": 0.25,
        "Sharpe Ratio": 1.4,
        "Max Drawdown": -0.05,
    }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_two_tickers",
        lambda *_args, **_kwargs: pair_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app._display_cointegration_result",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.run_walk_forward_optimisation",
        lambda *_args, **_kwargs: (
            {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5},
            expected_results,
            expected_metrics,
        ),
    )

    _, ticker_display, best_params, walk_forward_oos = (
        prepare_pairs_trading_strategy_without_optimisation(
            ("AAPL", "MSFT"),
            dates[0],
            dates[-1],
            {"window": [10], "entry_z_score": [2.0], "exit_z_score": [0.5]},
            optimise=True,
            walk_forward=True,
        )
    )

    assert ticker_display == "AAPL vs. MSFT"
    assert best_params == {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5}
    assert walk_forward_oos is not None
    assert walk_forward_oos[0].equals(expected_results)
    assert walk_forward_oos[1] == expected_metrics


def test_main_displays_and_saves_stitched_walk_forward_results(monkeypatch) -> None:
    """Verify app.main displays, plots, and saves stitched OOS results."""
    n_rows = 90
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    data = pl.DataFrame(
        {"Date": dates, "Close": [100.0 + i * 0.5 for i in range(n_rows)]}
    )
    param_ranges: dict[str, range | list[int | float]] = {
        "short_window": [3, 5],
        "long_window": [10, 15],
    }
    expected_params, expected_metrics, fold_results = walk_forward_optimise(
        data,
        "Moving Average Crossover",
        param_ranges,
        tickers="AAPL",
    )
    expected_results, _ = stitch_walk_forward_oos_results(fold_results)

    captured: dict[str, Any] = {}
    fake_session_state: dict[str, Any] = {}
    config = BacktestConfig(
        ticker="AAPL",
        start_date=dates[0],
        end_date=dates[-1],
        strategy_type="Moving Average Crossover",
        auto_select_tickers=False,
        optimise=True,
        walk_forward=True,
        strategy_params=param_ranges,
    )

    monkeypatch.setattr(app.st, "session_state", fake_session_state)
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_sidebar_inputs",
        lambda: (config, True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_full_company_name",
        lambda ticker: f"{ticker} Inc.",
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app._cached_evaluate_validation_period",
        lambda *_args, **_kwargs: pytest.fail(
            "Walk-forward flow must not re-evaluate only the final fold"
        ),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.display_performance_metrics",
        lambda metrics, _company: captured.__setitem__("metrics", metrics),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app._display_benchmark_comparison",
        lambda results, backtest_data, full_data=None: captured.update(
            benchmark_rows=len(results),
            benchmark_data_rows=len(backtest_data),
            benchmark_full_data_rows=len(full_data) if full_data is not None else 0,
        ),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.plot_equity_curve",
        lambda results, *_args, **_kwargs: captured.__setitem__(
            "equity_results", results
        ),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.plot_strategy_returns",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.display_returns_by_month",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.display_trade_ledger",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.display_historical_results",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.Backtester.save_results",
        lambda self, store=None: captured.update(
            saved_data_rows=len(self.data),
            saved_results_rows=len(self.results) if self.results is not None else 0,
            saved_params=self.strategy.get_parameters(),
        ),
    )

    app.main()

    assert captured["metrics"] == pytest.approx(expected_metrics, nan_ok=True)
    assert captured["equity_results"].equals(expected_results)
    assert captured["benchmark_rows"] == len(expected_results)
    assert captured["benchmark_data_rows"] == len(expected_results)
    assert captured["benchmark_full_data_rows"] == len(data)
    assert captured["saved_data_rows"] == len(expected_results)
    assert captured["saved_results_rows"] == len(expected_results)
    assert captured["saved_params"] == expected_params


def test_stitched_pairs_walk_forward_plots_varying_threshold_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify varying walk-forward pair thresholds render step traces."""
    fold_1_results = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
            "z_score": [-2.2, -0.3],
            "signal": [1.0, 0.0],
            "position_change": [1.0, -1.0],
            "gross_strategy_returns": [0.0, 0.02],
            "strategy_returns": [0.0, 0.019],
            "transaction_costs": [0.001, 0.001],
            "trade_turnover": [1.0, 1.0],
            "gross_cumulative_returns": [1.0, 1.02],
            "cumulative_returns": [1.0, 1.019],
            "cumulative_transaction_costs": [0.001, 0.002],
            "equity_curve": [100_000.0, 101_900.0],
        }
    )
    fold_2_results = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, 3), datetime.date(2020, 1, 4)],
            "z_score": [1.6, 0.2],
            "signal": [-1.0, 0.0],
            "position_change": [-1.0, 1.0],
            "gross_strategy_returns": [0.0, 0.01],
            "strategy_returns": [0.0, 0.009],
            "transaction_costs": [0.001, 0.001],
            "trade_turnover": [1.0, 1.0],
            "gross_cumulative_returns": [1.0, 1.01],
            "cumulative_returns": [1.0, 1.009],
            "cumulative_transaction_costs": [0.001, 0.002],
            "equity_curve": [100_000.0, 100_900.0],
        }
    )
    stitched, _ = optimiser.stitch_walk_forward_oos_results(
        [
            {
                "fold": 1,
                "params": {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5},
                "oos_results": fold_1_results,
            },
            {
                "fold": 2,
                "params": {"window": 20, "entry_z_score": 1.5, "exit_z_score": 0.2},
                "oos_results": fold_2_results,
            },
        ]
    )

    assert stitched["entry_z_score"].to_list() == [2.0, 2.0, 1.5, 1.5]
    assert stitched["exit_z_score"].to_list() == [0.5, 0.5, 0.2, 0.2]

    plotted_figs: list[object] = []
    monkeypatch.setattr(visualisation.st, "subheader", lambda *_a, **_k: None)
    monkeypatch.setattr(
        visualisation.st,
        "plotly_chart",
        lambda fig: plotted_figs.append(fig),
    )

    visualisation.plot_pairs_spread(
        stitched,
        "AAPL vs. MSFT",
        "Apple vs. Microsoft",
        entry_z_score=1.5,
        exit_z_score=0.2,
    )

    assert len(plotted_figs) == 1
    fig = plotted_figs[0]
    step_traces = [
        trace
        for trace in fig.data  # type: ignore[attr-defined]
        if trace.name in {"Entry (+)", "Entry (-)", "Exit (+)", "Exit (-)"}
    ]
    assert len(step_traces) == 4
    assert all(trace.line.shape == "hv" for trace in step_traces)
    assert list(step_traces[0].y) == [2.0, 2.0, 1.5, 1.5]


def test_walk_forward_status_callback_uses_custom_n_folds() -> None:
    """Verify walk-forward status message interpolates custom n_folds."""
    messages: list[str] = []

    data = pl.DataFrame(
        {
            "Date": [
                datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
                for i in range(60)
            ],
            "Close": [100.0 + i * 0.5 for i in range(60)],
        }
    )
    optimiser.walk_forward_optimise(
        data,
        "Moving Average Crossover",
        {"short_window": [3], "long_window": [10]},
        "AAPL",
        n_folds=3,
        status_callback=messages.append,
    )

    assert len(messages) == 1
    assert "3 expanding walk-forward folds" in messages[0]
    assert "stitched out-of-sample" in messages[0]


def test_main_persists_active_config_and_threads_trading_assumptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify form submit gates reruns and threads trading assumptions."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(15)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0 + i * 1.0 for i in range(15)]})
    fake_session_state: dict[str, Any] = {}
    sidebar_infos: list[str] = []
    saved_assumptions: list[tuple[float, float, float]] = []
    equity_starts: list[float] = []

    initial_config = BacktestConfig(
        ticker="AAPL",
        start_date=dates[0],
        end_date=dates[-1],
        strategy_type="Buy and Hold",
        auto_select_tickers=False,
        optimise=False,
        walk_forward=False,
        strategy_params={},
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
    )
    unsubmitted_config = BacktestConfig(
        ticker="MSFT",
        start_date=dates[0],
        end_date=dates[-1],
        strategy_type="Mean Reversion",
        auto_select_tickers=False,
        optimise=True,
        walk_forward=False,
        strategy_params={"window": [5, 10], "std_dev": [1.5, 2.0]},
        initial_capital=250_000.0,
        transaction_cost_bps=25.0,
        slippage_bps=15.0,
    )
    submitted_config = BacktestConfig(
        ticker="AAPL",
        start_date=dates[0],
        end_date=dates[-1],
        strategy_type="Buy and Hold",
        auto_select_tickers=False,
        optimise=False,
        walk_forward=False,
        strategy_params={},
        initial_capital=250_000.0,
        transaction_cost_bps=25.0,
        slippage_bps=15.0,
    )
    sidebar_responses = iter(
        [
            (initial_config, False),
            (unsubmitted_config, False),
            (submitted_config, True),
        ]
    )

    monkeypatch.setattr(app.st, "session_state", fake_session_state)
    monkeypatch.setattr(app.st.sidebar, "info", sidebar_infos.append)
    monkeypatch.setattr(app, "get_sidebar_inputs", lambda: next(sidebar_responses))
    monkeypatch.setattr(
        app,
        "load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: data,
    )
    monkeypatch.setattr(app, "get_full_company_name", lambda t: f"{t} Inc.")
    monkeypatch.setattr(
        app,
        "_run_optimisation_with_ui",
        lambda *_args, **_kwargs: pytest.fail(
            "Unsubmitted layout toggle must not trigger optimisation"
        ),
    )
    monkeypatch.setattr(app, "display_performance_metrics", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "_display_benchmark_comparison", lambda *_a, **_k: None)
    monkeypatch.setattr(
        app,
        "plot_equity_curve",
        lambda results, *_a, **_k: equity_starts.append(
            float(results["equity_curve"][0])
        ),
    )
    monkeypatch.setattr(app, "plot_strategy_returns", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "display_returns_by_month", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "display_trade_ledger", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "display_historical_results", lambda *_a, **_k: None)
    monkeypatch.setattr(
        app.Backtester,
        "save_results",
        lambda self, store=None: saved_assumptions.append(
            (
                self.initial_capital,
                self.transaction_cost_bps,
                self.slippage_bps,
            )
        ),
    )

    # First visit runs initial config automatically.
    app.main()
    assert fake_session_state["active_backtest_config"] == initial_config
    assert saved_assumptions == [(100_000.0, 5.0, 3.0)]

    # Toggling controls outside the form does not execute unsubmitted config.
    app.main()
    assert fake_session_state["active_backtest_config"] == initial_config
    assert len(sidebar_infos) == 1
    assert "Run Backtest" in sidebar_infos[0]
    assert saved_assumptions == [(100_000.0, 5.0, 3.0)]

    # Clicking Run Backtest applies the updated trading assumptions and saves.
    app.main()
    assert fake_session_state["active_backtest_config"] == submitted_config
    assert saved_assumptions == [
        (100_000.0, 5.0, 3.0),
        (250_000.0, 25.0, 15.0),
    ]
    assert equity_starts[0] == pytest.approx(100_000.0 * (1 - 0.0008))
    assert equity_starts[2] == pytest.approx(250_000.0 * (1 - 0.004))


@pytest.mark.parametrize(
    "start_date,end_date",
    [
        (datetime.date(2020, 6, 1), datetime.date(2020, 6, 1)),
        (datetime.date(2020, 6, 2), datetime.date(2020, 6, 1)),
    ],
)
def test_main_shows_error_and_returns_early_on_invalid_date_range(
    monkeypatch: pytest.MonkeyPatch,
    start_date: datetime.date,
    end_date: datetime.date,
) -> None:
    """Verify app.main rejects start_date >= end_date before fetching data."""
    errors: list[str] = []
    config = BacktestConfig(
        ticker="AAPL",
        start_date=start_date,
        end_date=end_date,
        strategy_type="Buy and Hold",
        auto_select_tickers=False,
        optimise=False,
        walk_forward=False,
        strategy_params={},
    )
    monkeypatch.setattr(app.st, "session_state", {})
    monkeypatch.setattr(app.st, "error", errors.append)
    monkeypatch.setattr(app, "get_sidebar_inputs", lambda: (config, True))
    monkeypatch.setattr(
        app,
        "_cached_load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: pytest.fail("Should not load data on invalid dates"),
    )

    app.main()

    assert len(errors) == 1
    assert "Start date must be earlier than end date" in errors[0]


def test_main_shows_error_on_invalid_fixed_strategy_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify app.main validates fixed strategy params submitted via form."""
    errors: list[str] = []
    config = BacktestConfig(
        ticker="AAPL",
        start_date=datetime.date(2020, 1, 1),
        end_date=datetime.date(2020, 12, 31),
        strategy_type="Moving Average Crossover",
        auto_select_tickers=False,
        optimise=False,
        walk_forward=False,
        strategy_params={"short_window": 40, "long_window": 20},
    )
    monkeypatch.setattr(app.st, "session_state", {})
    monkeypatch.setattr(app.st, "error", errors.append)
    monkeypatch.setattr(app, "get_sidebar_inputs", lambda: (config, True))
    monkeypatch.setattr(
        app,
        "_cached_load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: pytest.fail("Should not load data on invalid params"),
    )

    app.main()

    assert len(errors) == 1
    assert "short_window must be less than long_window" in errors[0]


def test_prepare_single_ticker_strategy_buy_and_hold_ignores_optimise_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pruned Buy and Hold branch does not run ticker optimisation."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0 + i for i in range(5)]})
    monkeypatch.setattr(
        app,
        "load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: data,
    )
    monkeypatch.setattr(
        app,
        "optimise_buy_and_hold_ticker",
        lambda *_args, **_kwargs: pytest.fail(
            "Buy and Hold manual prep should not run S&P 500 ticker selection"
        ),
    )

    returned_data, ticker_display, params, wf_oos = prepare_single_ticker_strategy(
        "AAPL",
        dates[0],
        dates[-1],
        "Buy and Hold",
        {},
        optimise=True,
    )

    assert returned_data.equals(data)
    assert ticker_display == "AAPL"
    assert params == {}
    assert wf_oos is None


def test_cached_is_same_company_uses_ciks_after_data_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Streamlit-cached CIKs avoid Yahoo fallback after cache clear."""
    fetch_calls = 0
    cik_map = {
        "GOOG": "0001652044",
        "GOOGL": "0001652044",
        "AAPL": "0000320193",
        "MSFT": "0000789019",
    }

    def mock_get_ciks() -> dict[str, str]:
        nonlocal fetch_calls
        fetch_calls += 1
        return dict(cik_map)

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_sp500_company_ciks",
        mock_get_ciks,
    )
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda *_args, **_kwargs: pytest.fail(
            "Should not query yfinance.Ticker when CIK mapping is available"
        ),
    )

    # First check populates the Streamlit CIK cache.
    assert app._cached_is_same_company("GOOG", "GOOGL") is True
    assert fetch_calls == 1

    # Clearing data.py caches must not trigger Yahoo fallback or a refetch.
    data.clear_data_caches()
    app._cached_is_same_company.clear()
    assert app._cached_is_same_company("GOOG", "GOOGL") is True
    assert app._cached_is_same_company("AAPL", "MSFT") is False
    assert fetch_calls == 1


def test_cached_cik_loader_does_not_stick_on_transient_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transient empty CIK lookups are cleared and retried later."""
    responses = iter([{}, {"GOOG": "0001652044", "GOOGL": "0001652044"}])
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_sp500_company_ciks",
        lambda: next(responses),
    )

    assert app._get_sp500_company_ciks_cached() == {}
    assert app._get_sp500_company_ciks_cached() == {
        "GOOG": "0001652044",
        "GOOGL": "0001652044",
    }
