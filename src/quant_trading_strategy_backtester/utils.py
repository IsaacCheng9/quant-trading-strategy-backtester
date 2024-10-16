"""
Contains utility code and constants used throughout the project.
"""

import logging

from sqlalchemy import create_engine

from quant_trading_strategy_backtester.models import Base

logger = logging.getLogger(__name__)
NUM_TOP_COMPANIES_ONE_TICKER = 100
NUM_TOP_COMPANIES_TWO_TICKERS = 20


def clear_database():
    """
    Utility function to clear all data from the database.

    This function drops all tables and recreates them, effectively clearing all data.
    Use with caution as this operation cannot be undone.

    Returns:
        None
    """
    # Create engine and connect to the database
    engine = create_engine("sqlite:///strategies.db")

    # Drop all tables
    Base.metadata.drop_all(engine)

    # Recreate all tables
    Base.metadata.create_all(engine)

    logger.info("Database cleared successfully.")


if __name__ == "__main__":
    clear_database()
