import datetime

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class StrategyModel(Base):
    """
    Represents a trading strategy in the database.

    Attributes:
        id: The primary key of the strategy.
        date_created: The date and time when the strategy was created.
        name: The name of the strategy.
        parameters: The parameters used for the strategy.
        total_return: The total return of the strategy.
        sharpe_ratio: The Sharpe ratio of the strategy.
        max_drawdown: The maximum drawdown of the strategy.
        tickers: The ticker(s) used in the strategy.
        start_date: The start date of the strategy backtest.
        end_date: The end date of the strategy backtest.
    """

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    date_created = Column(DateTime, default=datetime.datetime.now())
    name = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False)
    tickers = Column(JSON, nullable=False)  # Store as a JSON array of strings
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)


# Database setup
engine = create_engine("sqlite:///strategies.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
