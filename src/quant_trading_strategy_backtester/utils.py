"""Utility helpers and shared constants used throughout the project."""

import logging

from quant_trading_strategy_backtester.models import (
    DEFAULT_DATABASE_URL,
    Base,
    get_engine,
)

logger = logging.getLogger(__name__)
NUM_TOP_COMPANIES_ONE_TICKER = 100
NUM_TOP_COMPANIES_TWO_TICKERS = 20


def clear_database(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """Clear all stored strategy rows from the database.

    Drop all tables and recreate them on the target database engine.

    Args:
        database_url: SQLAlchemy database URL to reset.
    """
    engine = get_engine(database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    logger.info("Database cleared successfully.")


if __name__ == "__main__":
    clear_database()
