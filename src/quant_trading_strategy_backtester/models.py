"""SQLAlchemy ORM models and lazy database session helpers."""

import datetime
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

DEFAULT_DATABASE_URL = "sqlite:///strategies.db"

Base = declarative_base()

_ENGINES: dict[tuple[str, str | None], Engine] = {}
_SESSION_FACTORIES: dict[tuple[str, str | None], sessionmaker[OrmSession]] = {}


def _utc_now() -> datetime.datetime:
    """Return the current UTC timestamp."""
    return datetime.datetime.now(datetime.UTC)


def _database_cache_key(database_url: str) -> tuple[str, str | None]:
    """Return a cache key that scopes relative SQLite URLs by directory."""
    if (
        database_url.startswith("sqlite:///")
        and not database_url.startswith("sqlite:////")
        and database_url != "sqlite:///:memory:"
    ):
        return (database_url, str(Path.cwd()))
    return (database_url, None)


def get_engine(database_url: str = DEFAULT_DATABASE_URL) -> Engine:
    """Return a cached SQLAlchemy engine with tables initialised lazily.

    Args:
        database_url: SQLAlchemy database URL.

    Returns:
        An initialised SQLAlchemy Engine for the given URL.
    """
    cache_key = _database_cache_key(database_url)
    engine = _ENGINES.get(cache_key)
    if engine is None:
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        _ENGINES[cache_key] = engine
    return engine


def get_session_factory(
    database_url: str = DEFAULT_DATABASE_URL,
) -> sessionmaker[OrmSession]:
    """Return a cached SQLAlchemy session factory initialised lazily.

    Args:
        database_url: SQLAlchemy database URL.

    Returns:
        A sessionmaker bound to the lazily initialised engine for the URL.
    """
    cache_key = _database_cache_key(database_url)
    factory = _SESSION_FACTORIES.get(cache_key)
    if factory is None:
        engine = get_engine(database_url)
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
