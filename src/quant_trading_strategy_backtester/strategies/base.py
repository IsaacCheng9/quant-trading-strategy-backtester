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


class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    This class defines the interface for all trading strategies. Subclasses
    must implement the generate_signals method.
    """

    @abstractmethod
    def __init__(self, params: dict[str, Any]):
        self.params = params

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

    def get_parameters(self) -> dict[str, Any]:
        """
        Get the parameters of the strategy.

        Returns:
            A dictionary containing the strategy parameters.
        """
        return self.params
