"""
Defines abstract and concrete classes for trading strategies.

It provides a framework for creating and implementing various trading
strategies. These templates can be used as a template to develop and test
different quantitative trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


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
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.

        Args:
            data: Market data used to generate trading signals.

        Returns:
            Trading signals generated by the strategy.
        """
        raise NotImplementedError("Method 'generate_signals' must be implemented.")


class MeanReversionStrategy(Strategy):
    """
    Implements the mean reversion strategy, which is based on the assumption
    that asset prices tend to revert to their mean over time. Prices are
    assumed to follow a normal distribution over time, and extreme deviations
    from the mean are statistically less likely to persist.  This strategy uses
    a moving average and standard deviation to create upper and lower price
    bands.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        # The number of days to calculate the moving average and standard
        # deviation.
        self.window = int(params["window"])
        # The number of standard deviations to use for the price bands. This
        # sets the upper and lower bands for buy and sell signals
        # (mean +/- std_dev).
        self.std_dev = float(params["std_dev"])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals for the given data.

        Generates a buy signal (1) when the price falls below the lower band,
        and generates a sell signal (-1) when the price rises above the upper
        band. The strategy assumes mean reversion will occur.

        Args:
            data: A DataFrame containing the price data. Must have a 'Close'
                  column.

        Returns:
            A DataFrame containing the generated trading signals. Columns
            include 'signal', 'mean', 'std', 'upper_band', 'lower_band', and
            'positions'.
        """
        signals = pd.DataFrame(index=data.index)
        signals["mean"] = (
            data["Close"].rolling(window=self.window, min_periods=1).mean()
        )
        signals["std"] = data["Close"].rolling(window=self.window, min_periods=1).std()
        # Avoid division by zero.
        signals["std"] = signals["std"].replace(0, np.nan)

        signals["upper_band"] = signals["mean"] + (self.std_dev * signals["std"])
        signals["lower_band"] = signals["mean"] - (self.std_dev * signals["std"])

        signals["signal"] = 0.0
        # Buy signal
        signals.loc[data["Close"] < signals["lower_band"], "signal"] = 1.0
        # Sell signal
        signals.loc[data["Close"] > signals["upper_band"], "signal"] = -1.0

        # Fill NaN values with 0 (no signal)
        signals["signal"] = signals["signal"].fillna(0)
        signals["positions"] = signals["signal"].diff()

        return signals


class MovingAverageCrossoverStrategy(Strategy):
    """
    Implements the moving average crossover strategy, which is based on the
    crossover of short-term and long-term moving averages of the closing price.
    This strategy aims to identify and follows market trends. The short-term
    moving average is more responsive to price changes, while the long-term
    moving average represents the overall trend direction.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        # The number of days for the short-term and long-term moving average.
        self.short_window = int(params["short_window"])
        self.long_window = int(params["long_window"])

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
        # If the short-term moving average is above the long-term moving
        # average, generate a buy signal.
        signals.loc[signals["short_mavg"] > signals["long_mavg"], "signal"] = 1.0
        # If the short-term moving average is below the long-term moving
        # average, generate a sell signal by setting all non-buy signals to -1.
        signals["positions"] = signals["signal"].diff()

        return signals


class PairsTradingStrategy(Strategy):
    """
    Implements a pairs trading strategy for two correlated financial
    instruments, which is based on the assumption that the historical
    relationship between two assets will continue. This is a market-neutral
    strategy that exploits temporary mispricings in this relationship. It
    calculates a spread between the two assets and uses z-scores to determine
    when to enter and exit positions.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        # The lookback period for calculating the rolling mean and standard
        # deviation of the spread.
        self.window = int(params["window"])
        # The score threshold for entering a trade.
        self.entry_z_score = float(params["entry_z_score"])
        # The z-score threshold for exiting a trade.
        self.exit_z_score = float(params["exit_z_score"])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the pairs trading strategy.

        Calculates the spread between two assets, computes the z-score of the
        spread, and generates trading signals based on the entry and exit
        z-score thresholds.

        Trading Logic:
            - When z-score > entry_z_score:
                - Short asset 1, long asset 2 (signal = -1)
            - When z-score < -entry_z_score:
                - Long asset 1, short asset 2 (signal = 1)
            - When |z-score| < exit_z_score:
                - Close positions (signal = 0)

        Args:
            data: A DataFrame containing the price data. Must have 'Close_1'
                  and 'Close_2' columns representing the closing prices of the
                  two assets.

        Returns:
            A DataFrame containing the generated trading signals.
            Columns include:
            - 'spread': The price difference between the two assets.
            - 'z_score': The standardized score of the spread.
            - 'signal': The trading signal (-1, 0, or 1).
            - 'positions': The change in position from the previous period.
        """
        if "Close_1" not in data.columns or "Close_2" not in data.columns:
            raise ValueError("Data must contain 'Close_1' and 'Close_2' columns")

        signals = pd.DataFrame(index=data.index)

        # Calculate the spread.
        signals["spread"] = data["Close_1"] - data["Close_2"]

        # Calculate z-score of the spread.
        signals["z_score"] = (
            signals["spread"] - signals["spread"].rolling(window=self.window).mean()
        ) / signals["spread"].rolling(window=self.window).std()

        # Generate trading signals.
        signals["signal"] = 0.0
        # Short stock 1, long stock 2.
        signals.loc[signals["z_score"] > self.entry_z_score, "signal"] = -1
        # Long stock 1, short stock 2.
        signals.loc[signals["z_score"] < -self.entry_z_score, "signal"] = 1
        # Close positions
        signals.loc[signals["z_score"].abs() < self.exit_z_score, "signal"] = 0

        signals["positions"] = signals["signal"].diff()

        return signals
