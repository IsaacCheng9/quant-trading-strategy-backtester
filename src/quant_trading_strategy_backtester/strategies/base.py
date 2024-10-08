"""
Defines abstract and concrete classes for trading strategies.

It provides a framework for creating and implementing various trading
strategies. These templates can be used as a template to develop and test
different quantitative trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

TRADING_STRATEGIES = [
    "Buy and Hold",
    "Mean Reversion",
    "Moving Average Crossover",
    "Pairs Trading",
]


class Strategy(ABC):
    """
    Abstract base class for a trading strategy.

    This class defines the interface for all trading strategies. Subclasses
    must implement the generate_signals method.
    """

    # TODO: Consider setting the params back to indivdual attributes. Would this be easier to read?
    @abstractmethod
    def __init__(self, params: dict[str, Any]):
        raise NotImplementedError("Method '__init__' must be implemented.")

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on the input data.

        Args:
            data: Market data used to generate trading signals.

        Returns:
            Trading signals generated by the strategy.
        """
        raise NotImplementedError("Method 'generate_signals' must be implemented.")
