"""SQLAlchemy ORM models and lazy database session helpers."""

import datetime
import threading
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DATABASE_URL = "sqlite:///strategies.db"

Base = declarative_base()

_CACHE_LOCK = threading.Lock()
_ENGINES: dict[tuple[str, str | None], Engine] = {}
_SESSION_FACTORIES: dict[tuple[str, str | None], sessionmaker[OrmSession]] = {}


def _utc_now() -> datetime.datetime:
    """Return the current UTC timestamp."""
    return datetime.datetime.now(datetime.UTC)


def _is_in_memory_sqlite_url(database_url: str) -> bool:
    """Return whether a SQLAlchemy URL refers to an in-memory SQLite DB."""
    return database_url in {"sqlite://", "sqlite:///:memory:"} or (
        database_url.startswith("sqlite:") and ":memory:" in database_url
    )


def _database_cache_key(database_url: str) -> tuple[str, str | None]:
    """Return a cache key that scopes relative SQLite URLs by directory."""
    if database_url.startswith("sqlite:///") and not database_url.startswith(
        "sqlite:////"
    ):
        return (database_url, str(Path.cwd()))
    return (database_url, None)


def _create_initialised_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine and initialise tables for the given URL."""
    if _is_in_memory_sqlite_url(database_url):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def _get_or_create_engine_locked(database_url: str) -> Engine:
    """Return a cached engine while holding _CACHE_LOCK."""
    cache_key = _database_cache_key(database_url)
    engine = _ENGINES.get(cache_key)
    if engine is None:
        engine = _create_initialised_engine(database_url)
        _ENGINES[cache_key] = engine
    return engine


def get_engine(database_url: str = DEFAULT_DATABASE_URL) -> Engine:
    """Return a SQLAlchemy engine with tables initialised lazily.

    Persistent URLs are cached safely across threads. In-memory SQLite URLs
    bypass the global cache so independent stores remain isolated.

    Args:
        database_url: SQLAlchemy database URL.

    Returns:
        An initialised SQLAlchemy Engine for the given URL.
    """
    if _is_in_memory_sqlite_url(database_url):
        return _create_initialised_engine(database_url)

    with _CACHE_LOCK:
        return _get_or_create_engine_locked(database_url)


def get_session_factory(
    database_url: str = DEFAULT_DATABASE_URL,
) -> sessionmaker[OrmSession]:
    """Return a SQLAlchemy session factory initialised lazily.

    Persistent URLs are cached safely across threads. In-memory SQLite URLs
    bypass the global cache so independent stores remain isolated.

    Args:
        database_url: SQLAlchemy database URL.

    Returns:
        A sessionmaker bound to the lazily initialised engine for the URL.
    """
    if _is_in_memory_sqlite_url(database_url):
        return sessionmaker(bind=_create_initialised_engine(database_url))

    cache_key = _database_cache_key(database_url)
    with _CACHE_LOCK:
        factory = _SESSION_FACTORIES.get(cache_key)
        if factory is None:
            engine = _get_or_create_engine_locked(database_url)
            factory = sessionmaker(bind=engine)
            _SESSION_FACTORIES[cache_key] = factory
        return factory


class StrategyModel(Base):
    """Represent a trading strategy backtest result in the database.

    Attributes:
        id: Primary key of the strategy record.
        date_created: UTC timestamp when the strategy was saved.
        name: Name of the strategy class.
        parameters: JSON-serialisable strategy parameters.
        total_return: Total return of the strategy.
        sharpe_ratio: Annualised Sharpe ratio, or None when undefined.
        max_drawdown: Maximum drawdown of the strategy.
        tickers: Ticker symbol or list of ticker symbols used.
        start_date: Start date of the strategy backtest.
        end_date: End date of the strategy backtest.
    """

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    date_created = Column(DateTime, default=_utc_now)
    name = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False)
    tickers = Column(JSON, nullable=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
