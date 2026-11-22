"""Tests for Streamlit app helpers."""

import datetime

import polars as pl
import pytest

from quant_trading_strategy_backtester import app, visualisation
from quant_trading_strategy_backtester.app import (
    _build_save_key,
    _display_benchmark_comparison,
    _format_historical_sharpe,
    _get_benchmark_date_range,
    _get_prior_date,
    _get_strategy_store,
    _streamlit_progress,
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
