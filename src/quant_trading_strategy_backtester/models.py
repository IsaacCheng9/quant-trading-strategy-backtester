from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    date_created = Column(DateTime, default=datetime.datetime.now())
    name = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)


# Database setup
engine = create_engine("sqlite:///strategies.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
