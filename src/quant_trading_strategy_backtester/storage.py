"""Persistence layer for storing and retrieving backtest strategy records."""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

from sqlalchemy import JSON
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from quant_trading_strategy_backtester.models import (
    DEFAULT_DATABASE_URL,
    StrategyModel,
    get_session_factory,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def _normalise_datetime(value: datetime | None) -> datetime:
    """Normalise a datetime to timezone-aware UTC."""
    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalise_sharpe_ratio(value: float | None) -> float | None:
    """Return a finite Sharpe ratio float, or None when undefined."""
    if value is None:
        return None
    float_value = float(value)
    if not math.isfinite(float_value):
        return None
    return float_value


def _normalise_json_value(value: Any) -> Any:
    """Recursively convert values into canonical JSON-compatible objects."""
    if hasattr(value, "item") and not isinstance(value, (dict, list, tuple)):
        value = value.item()

    if isinstance(value, dict):
        return {
            str(key): _normalise_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _normalise_parameters(parameters: Any) -> dict[str, Any]:
    """Decode legacy JSON strings and return canonical sorted parameters."""
    value = parameters
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {}
        value = parsed if isinstance(parsed, dict) else {}

    if not isinstance(value, dict):
        return {}

    return cast(dict[str, Any], _normalise_json_value(value))


def _normalise_tickers(tickers: Any) -> str | list[str] | None:
    """Decode legacy JSON strings and return normalised tickers."""
    value = tickers
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        if parsed is None or isinstance(parsed, (str, list)):
            value = parsed

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class StrategyRecord:
    """Represent a persisted strategy backtest summary.

    Attributes:
        name: Strategy class name.
        parameters: Strategy parameters dictionary.
        total_return: Net total return over the backtest period.
        sharpe_ratio: Annualised Sharpe ratio, or None when undefined.
        max_drawdown: Maximum drawdown over the backtest period.
        tickers: Ticker symbol or list of symbols tested.
        start_date: Inclusive first date of the backtest period.
        end_date: Inclusive last date of the backtest period.
        date_created: UTC timestamp when the record was created.
    """

    name: str
    parameters: dict[str, Any]
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    tickers: str | list[str] | None
    start_date: date
    end_date: date
    date_created: datetime = field(default_factory=_utc_now)


class StrategyStore(Protocol):
    """Protocol for persisting and listing strategy backtest records."""

    def save(self, record: StrategyRecord) -> bool:
        """Persist a strategy record if an identical run does not exist.

        Args:
            record: Strategy record to persist.

        Returns:
            True if saved, False if skipped as a duplicate.
        """
        ...

    def list_recent(self) -> list[StrategyRecord]:
        """Return stored strategy records ordered newest first."""
        ...


class SQLiteStrategyStore:
    """Persist strategy records in SQLite using lazy SQLAlchemy sessions."""

    def __init__(
        self,
        session_factory: sessionmaker[OrmSession] | Callable[[], Any] | None = None,
        database_url: str = DEFAULT_DATABASE_URL,
    ) -> None:
        self._session_factory = session_factory
        self._database_url = database_url

    def _get_session_factory(self) -> sessionmaker[OrmSession] | Callable[[], Any]:
        """Resolve the session factory lazily on first use."""
        if self._session_factory is None:
            self._session_factory = get_session_factory(self._database_url)
        return self._session_factory

    def save(self, record: StrategyRecord) -> bool:
        """Save a strategy record to SQLite unless a duplicate exists.

        Args:
            record: Strategy record to store.

        Returns:
            True if a new row was inserted, False if skipped as a duplicate.

        Raises:
            ValueError: If committing the database transaction fails.
        """
        normalised_params = _normalise_parameters(record.parameters)
        normalised_tickers = _normalise_tickers(record.tickers)
        normalised_sharpe = _normalise_sharpe_ratio(record.sharpe_ratio)
        created_at = _normalise_datetime(record.date_created)

        session = self._get_session_factory()()
        try:
            candidates = (
                session.query(StrategyModel)
                .filter_by(
                    name=record.name,
                    start_date=record.start_date,
                    end_date=record.end_date,
                )
                .all()
            )
            for existing in candidates:
                if (
                    _normalise_parameters(existing.parameters) == normalised_params
                    and _normalise_tickers(existing.tickers) == normalised_tickers
                ):
                    logger.info(
                        "Strategy %s with same parameters and tickers already "
                        "exists. Skipping save.",
                        record.name,
                    )
                    return False

            # Use JSON.NULL when tickers is None so legacy NOT NULL columns
            # still accept rows without losing None round-tripping.
            stored_tickers: Any = (
                JSON.NULL if normalised_tickers is None else normalised_tickers
            )
            new_strategy = StrategyModel(
                date_created=created_at,
                name=record.name,
                parameters=normalised_params,
                total_return=float(record.total_return),
                sharpe_ratio=normalised_sharpe,
                max_drawdown=float(record.max_drawdown),
                tickers=stored_tickers,
                start_date=record.start_date,
                end_date=record.end_date,
            )
            session.add(new_strategy)
            session.commit()
            logger.info("Strategy %s saved successfully.", record.name)
            return True
        except Exception as exc:
            session.rollback()
            raise ValueError(f"Failed to save strategy results: {exc!s}") from exc
        finally:
            session.close()

    def list_recent(self) -> list[StrategyRecord]:
        """Return all stored strategy records ordered newest first."""
        session = self._get_session_factory()()
        try:
            rows = (
                session.query(StrategyModel)
                .order_by(
                    StrategyModel.date_created.desc(),
                    StrategyModel.id.desc(),
                )
                .all()
            )
            return [self._row_to_record(row) for row in rows]
        finally:
            session.close()

    @staticmethod
    def _row_to_record(row: StrategyModel) -> StrategyRecord:
        """Convert an ORM StrategyModel row into a StrategyRecord."""
        return StrategyRecord(
            name=str(row.name),
            parameters=_normalise_parameters(row.parameters),
            total_return=float(cast(float, row.total_return)),
            sharpe_ratio=_normalise_sharpe_ratio(cast(float | None, row.sharpe_ratio)),
            max_drawdown=float(cast(float, row.max_drawdown)),
            tickers=_normalise_tickers(row.tickers),
            start_date=cast(date, row.start_date),
            end_date=cast(date, row.end_date),
            date_created=_normalise_datetime(cast(datetime | None, row.date_created)),
        )


class InMemoryStrategyStore:
    """Persist strategy records in a plain Python list."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records: list[dict[str, Any]] = [] if records is None else records

    def save(self, record: StrategyRecord) -> bool:
        """Save a strategy record to the backing list unless duplicated.

        Args:
            record: Strategy record to store.

        Returns:
            True if appended, False if skipped as a duplicate.
        """
        normalised_params = _normalise_parameters(record.parameters)
        normalised_tickers = _normalise_tickers(record.tickers)
        normalised_sharpe = _normalise_sharpe_ratio(record.sharpe_ratio)
        created_at = _normalise_datetime(record.date_created)

        for existing in self._records:
            if (
                existing.get("name") == record.name
                and _normalise_parameters(existing.get("parameters"))
                == normalised_params
                and _normalise_tickers(existing.get("tickers")) == normalised_tickers
                and existing.get("start_date") == record.start_date
                and existing.get("end_date") == record.end_date
            ):
                logger.info(
                    "Strategy %s with same parameters and tickers already "
                    "exists in memory. Skipping save.",
                    record.name,
                )
                return False

        self._records.append(
            {
                "date_created": created_at,
                "name": record.name,
                "parameters": normalised_params,
                "total_return": float(record.total_return),
                "sharpe_ratio": normalised_sharpe,
                "max_drawdown": float(record.max_drawdown),
                "tickers": normalised_tickers,
                "start_date": record.start_date,
                "end_date": record.end_date,
            }
        )
        logger.info("Strategy %s saved to in-memory store.", record.name)
        return True

    def list_recent(self) -> list[StrategyRecord]:
        """Return stored in-memory strategy records ordered newest first."""
        records = [
            StrategyRecord(
                name=str(entry["name"]),
                parameters=_normalise_parameters(entry.get("parameters")),
                total_return=float(entry["total_return"]),
                sharpe_ratio=_normalise_sharpe_ratio(entry.get("sharpe_ratio")),
                max_drawdown=float(entry["max_drawdown"]),
                tickers=_normalise_tickers(entry.get("tickers")),
                start_date=cast(date, entry["start_date"]),
                end_date=cast(date, entry["end_date"]),
                date_created=_normalise_datetime(entry.get("date_created")),
            )
            for entry in reversed(self._records)
        ]
        records.sort(key=lambda item: item.date_created, reverse=True)
        return records
