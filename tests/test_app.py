"""Tests for Streamlit app helpers."""

import datetime

import polars as pl
import pytest

from quant_trading_strategy_backtester import app
from quant_trading_strategy_backtester.app import (
    _build_save_key,
    _format_historical_sharpe,
    _get_benchmark_date_range,
    _get_strategy_store,
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
