"""Tests for Streamlit app helpers."""

import datetime
from typing import Any

import polars as pl
import pytest

from quant_trading_strategy_backtester import app, visualisation
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


def test_get_benchmark_date_range_uses_exclusive_yahoo_end_date() -> None:
    """Verify benchmark loading includes the displayed final backtest date."""
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
    assert end_date == datetime.date(2020, 1, 4)


def test_build_save_key_changes_with_dates_and_modes() -> None:
    """Verify save keys distinguish dates, modes, and canonicalised params."""
    base_kwargs = {
        "strategy_type": "Moving Average Crossover",
        "strategy_params": {"short_window": 5, "long_window": 20},
        "tickers": "AAPL",
        "start_date": datetime.date(2020, 1, 1),
        "end_date": datetime.date(2020, 12, 31),
        "optimise": False,
        "walk_forward": False,
        "auto_select_tickers": False,
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
    assert end_date == datetime.date(2020, 1, 8)


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

    assert loaded_range == [(datetime.date(2020, 1, 1), datetime.date(2020, 1, 5))]
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

    assert captured["load_two_tickers"] is app._cached_load_yfinance_data_two_tickers
    assert captured["same_company_check"] is app._cached_is_same_company


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

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_user_inputs_except_strategy_params",
        lambda: ("AAPL", dates[0], dates[-1], "Moving Average Crossover", False),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_user_inputs_for_strategy_params",
        lambda *_args, **_kwargs: (True, True, param_ranges),
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
        lambda results, backtest_data: captured.update(
            benchmark_rows=len(results), benchmark_data_rows=len(backtest_data)
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
        lambda: None,
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
    assert captured["saved_data_rows"] == len(expected_results)
    assert captured["saved_results_rows"] == len(expected_results)
    assert captured["saved_params"] == expected_params
