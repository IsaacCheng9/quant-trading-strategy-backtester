"""
Defines abstract and concrete classes for trading strategies.

It provides a framework for creating and implementing various trading
strategies. These templates can be used as a template to develop and test
different quantitative trading strategies.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for a trading strategy.

    This class defines the interface for all trading strategies. Subclasses
    must implement the generate_signals method.
    """

    @abstractmethod
    def generate_signals(self, data) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.

        Args:
            data: Market data used to generate trading signals.

        Returns:
            Trading signals generated by the strategy.
        """
        raise NotImplementedError("Method 'generate_signals' must be implemented.")


class MovingAverageCrossoverStrategy(Strategy):
    """
    Generates buy and sell signals based on the crossover of short-term and
    long-term moving averages of the closing price.

    Attributes:
        short_window: The number of periods for the short-term moving average.
        long_window: The number of periods for the long-term moving average.
    """

    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals for the given data.

        A buy signal (1) is generated when the short-term moving average
        crosses above the long-term moving average. The strategy maintains the
        position until the short-term moving average crosses below the
        long-term moving average.

        Args:
            data: A DataFrame containing the price data. Must have a 'Close'
                  column.

        Returns:
            A DataFrame containing the generated trading signals. Columns
            include 'signal', 'short_mavg', 'long_mavg', and 'positions'.
        """
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0.0
        signals["short_mavg"] = (
            data["Close"]
            .rolling(window=self.short_window, min_periods=1, center=False)
            .mean()
        )
        signals["long_mavg"] = (
            data["Close"]
            .rolling(window=self.long_window, min_periods=1, center=False)
            .mean()
        )
        signals.loc[signals["short_mavg"] > signals["long_mavg"], "signal"] = 1.0
        signals["positions"] = signals["signal"].diff()

        return signals
