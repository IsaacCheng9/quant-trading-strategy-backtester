"""Tests for lazy database setup and strategy persistence stores."""

import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import sqlalchemy

from quant_trading_strategy_backtester import models
from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.models import StrategyModel
from quant_trading_strategy_backtester.storage import (
    InMemoryStrategyStore,
    SQLiteStrategyStore,
    StrategyRecord,
)
from quant_trading_strategy_backtester.strategies.buy_and_hold import BuyAndHoldStrategy
from quant_trading_strategy_backtester.utils import clear_database


def _make_record(
    *,
    name: str = "BuyAndHoldStrategy",
    parameters: dict[str, object] | None = None,
    total_return: float = 0.12,
    sharpe_ratio: float | None = 1.25,
    max_drawdown: float = -0.05,
    tickers: str | list[str] | None = "AAPL",
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2020, 12, 31),
    date_created: datetime | None = None,
) -> StrategyRecord:
    """Create a StrategyRecord with sensible defaults for testing."""
    return StrategyRecord(
        name=name,
        parameters={} if parameters is None else parameters,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        date_created=(
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            if date_created is None
            else date_created
        ),
    )


def test_importing_core_modules_creates_no_db_or_streamlit_import(
    tmp_path: Path,
) -> None:
    """Verify importing models, storage, and backtester stays lazy and pure."""
    code = (
        "import sys; "
        "import quant_trading_strategy_backtester.models; "
        "import quant_trading_strategy_backtester.storage; "
        "import quant_trading_strategy_backtester.backtester; "
        "assert 'streamlit' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "strategies.db").exists()


def test_backtester_construction_and_run_do_not_touch_database(
    monkeypatch: pytest.MonkeyPatch,
    mock_polars_data: pl.DataFrame,
) -> None:
    """Verify constructing and running Backtester allocates no DB session."""

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Database access should not occur during backtest")

    monkeypatch.setattr(models, "get_session_factory", _fail)
    monkeypatch.setattr(models, "get_engine", _fail)
    monkeypatch.setattr(sqlalchemy, "create_engine", _fail)

    backtester = Backtester(
        mock_polars_data,
        BuyAndHoldStrategy({}),
        tickers="AAPL",
    )
    results = backtester.run()
    metrics = backtester.get_performance_metrics()
    record = backtester.build_strategy_record()

    assert not results.is_empty()
    assert metrics is not None
    assert record.name == "BuyAndHoldStrategy"
    assert record.tickers == "AAPL"


def test_sqlite_store_saves_distinct_tickers_on_same_dates(
    sqlite_store: SQLiteStrategyStore,
) -> None:
    """Verify Buy and Hold saves MSFT after AAPL over the same date range."""
    aapl_record = _make_record(
        tickers="AAPL",
        date_created=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
    )
    msft_record = _make_record(
        tickers="MSFT",
        date_created=datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
    )

    assert sqlite_store.save(aapl_record) is True
    assert sqlite_store.save(msft_record) is True

    recent = sqlite_store.list_recent()
    assert len(recent) == 2
    assert [item.tickers for item in recent] == ["MSFT", "AAPL"]


def test_sqlite_store_deduplicates_regardless_of_param_key_order(
    sqlite_store: SQLiteStrategyStore,
) -> None:
    """Verify duplicate check ignores parameter dictionary key ordering."""
    first = _make_record(
        name="MovingAverageCrossoverStrategy",
        parameters={"short_window": 5, "long_window": 20},
        tickers="AAPL",
    )
    second = _make_record(
        name="MovingAverageCrossoverStrategy",
        parameters={"long_window": 20, "short_window": 5},
        tickers="AAPL",
    )

    assert sqlite_store.save(first) is True
    assert sqlite_store.save(second) is False
    assert len(sqlite_store.list_recent()) == 1


def test_sqlite_store_stores_native_json_without_double_encoding(
    sqlite_store: SQLiteStrategyStore,
    session_factory,
) -> None:
    """Verify parameters and tickers are stored as native JSON objects."""
    record = _make_record(
        name="PairsTradingStrategy",
        parameters={"z_exit": 0.5, "window": 20},
        tickers=["AAPL", "MSFT"],
    )

    assert sqlite_store.save(record) is True

    with session_factory() as session:
        row = session.query(StrategyModel).one()
        assert isinstance(row.parameters, dict)
        assert row.parameters == {"window": 20, "z_exit": 0.5}
        assert list(row.parameters.keys()) == ["window", "z_exit"]
        assert isinstance(row.tickers, list)
        assert row.tickers == ["AAPL", "MSFT"]


def test_sqlite_store_reads_and_deduplicates_legacy_double_encoded_rows(
    sqlite_store: SQLiteStrategyStore,
    session_factory,
) -> None:
    """Verify legacy rows with json.dumps strings decode cleanly."""
    with session_factory() as session:
        session.add(
            StrategyModel(
                date_created=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
                name="PairsTradingStrategy",
                parameters=json.dumps({"window": 20, "entry_z_score": 2.0}),
                total_return=0.15,
                sharpe_ratio=1.1,
                max_drawdown=-0.04,
                tickers=json.dumps(["AAPL", "MSFT"]),
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
            )
        )
        session.commit()

    recent = sqlite_store.list_recent()
    assert len(recent) == 1
    assert recent[0].parameters == {"entry_z_score": 2.0, "window": 20}
    assert recent[0].tickers == ["AAPL", "MSFT"]

    duplicate = _make_record(
        name="PairsTradingStrategy",
        parameters={"entry_z_score": 2.0, "window": 20},
        tickers=["AAPL", "MSFT"],
    )
    assert sqlite_store.save(duplicate) is False


def test_sqlite_store_normalises_non_finite_sharpe_and_handles_none_tickers(
    sqlite_store: SQLiteStrategyStore,
) -> None:
    """Verify NaN Sharpe becomes None, 0.0 stays 0.0, and None tickers work."""
    nan_record = _make_record(
        tickers=None,
        sharpe_ratio=float("nan"),
        date_created=datetime(2024, 1, 1, 9, 0, 0, tzinfo=UTC),
    )
    zero_record = _make_record(
        tickers="NVDA",
        sharpe_ratio=0.0,
        date_created=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
    )

    assert sqlite_store.save(nan_record) is True
    assert sqlite_store.save(zero_record) is True

    recent = sqlite_store.list_recent()
    assert len(recent) == 2
    assert recent[0].tickers == "NVDA"
    assert recent[0].sharpe_ratio == 0.0
    assert recent[1].tickers is None
    assert recent[1].sharpe_ratio is None


def test_in_memory_store_deduplicates_and_orders_newest_first() -> None:
    """Verify InMemoryStrategyStore wraps a list, deduplicates, and sorts."""
    backing_list: list[dict[str, object]] = []
    store = InMemoryStrategyStore(backing_list)

    older = _make_record(
        name="MeanReversionStrategy",
        parameters={"window": 10, "std_dev": 2.0},
        tickers="AAPL",
        sharpe_ratio=float("nan"),
        date_created=datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC),
    )
    duplicate = _make_record(
        name="MeanReversionStrategy",
        parameters={"std_dev": 2.0, "window": 10},
        tickers="AAPL",
        sharpe_ratio=1.0,
        date_created=datetime(2024, 1, 1, 9, 0, 0, tzinfo=UTC),
    )
    newer = _make_record(
        name="MeanReversionStrategy",
        parameters={"window": 10, "std_dev": 2.0},
        tickers="MSFT",
        sharpe_ratio=0.0,
        date_created=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
    )

    assert store.save(older) is True
    assert store.save(duplicate) is False
    assert store.save(newer) is True
    assert len(backing_list) == 2

    recent = store.list_recent()
    assert [item.tickers for item in recent] == ["MSFT", "AAPL"]
    assert recent[0].sharpe_ratio == 0.0
    assert recent[1].sharpe_ratio is None


def test_clear_database_resets_target_database_only(tmp_path: Path) -> None:
    """Verify clear_database drops and recreates tables for a custom URL."""
    db_path = tmp_path / "custom.db"
    db_url = f"sqlite:///{db_path}"
    store = SQLiteStrategyStore(database_url=db_url)

    assert store.save(_make_record(tickers="AAPL")) is True
    assert len(store.list_recent()) == 1

    clear_database(database_url=db_url)

    assert store.list_recent() == []
    assert db_path.exists()
