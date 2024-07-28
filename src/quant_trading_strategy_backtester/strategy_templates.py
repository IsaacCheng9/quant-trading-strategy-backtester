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

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
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
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("short_mavg", pl.Float64),
                    ("long_mavg", pl.Float64),
                    ("signal", pl.Float64),
                    ("positions", pl.Float64),
                ]
            )

        signals = data.select([pl.col("Date"), pl.col("Close")])
        signals = signals.with_columns(
            [
                pl.col("Close")
                .rolling_mean(window_size=self.window, min_periods=1)
                .alias("mean"),
                pl.col("Close")
                .rolling_std(window_size=self.window, min_periods=1)
                .alias("std"),
            ]
        )
        # Avoid division by zero by replacing 0s with NaN.
        signals = signals.with_columns(
            [
                pl.when(pl.col("std") == 0)
                .then(pl.lit(float("nan")))
                .otherwise(pl.col("std"))
                .alias("std")
            ]
        )

        signals = signals.with_columns(
            [
                (pl.col("mean") + (self.std_dev * pl.col("std"))).alias("upper_band"),
                (pl.col("mean") - (self.std_dev * pl.col("std"))).alias("lower_band"),
            ]
        )

        signals = signals.with_columns(
            [
                # Buy signal
                pl.when(pl.col("Close") < pl.col("lower_band"))
                .then(1.0)
                # Sell signal
                .when(pl.col("Close") > pl.col("upper_band"))
                .then(-1.0)
                .otherwise(0.0)
                .alias("signal")
            ]
        )

        signals = signals.with_columns(
            [pl.col("signal").diff().fill_null(0).alias("positions")]
        )

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

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
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
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("short_mavg", pl.Float64),
                    ("long_mavg", pl.Float64),
                    ("signal", pl.Float64),
                    ("positions", pl.Float64),
                ]
            )

        signals = data.select([pl.col("Date"), pl.col("Close")])
        signals = signals.with_columns(
            [
                pl.col("Close")
                .rolling_mean(window_size=self.short_window, min_periods=1)
                .alias("short_mavg"),
                pl.col("Close")
                .rolling_mean(window_size=self.long_window, min_periods=1)
                .alias("long_mavg"),
                pl.lit(0.0).alias("signal"),
            ]
        )

        signals = signals.with_columns(
            [
                # If the short-term moving average is above the long-term moving
                # average, generate a buy signal.
                pl.when(pl.col("short_mavg") > pl.col("long_mavg"))
                .then(1.0)
                .otherwise(0.0)
                .alias("signal")
            ]
        )

        # If the short-term moving average is below the long-term moving
        # average, generate a sell signal by setting all non-buy signals to -1.
        signals = signals.with_columns(
            [pl.col("signal").diff().fill_null(0.0).alias("positions")]
        )

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

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
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
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("short_mavg", pl.Float64),
                    ("long_mavg", pl.Float64),
                    ("signal", pl.Float64),
                    ("positions", pl.Float64),
                ]
            )
        if "Close_1" not in data.columns or "Close_2" not in data.columns:
            raise ValueError("Data must contain 'Close_1' and 'Close_2' columns")

        signals = data.select(
            [
                pl.col("Date"),
                pl.col("Close_1"),
                pl.col("Close_2"),
                (pl.col("Close_1") - pl.col("Close_2")).alias("spread"),
            ]
        )

        # Calculate rolling mean and std, handling potential division by zero
        signals = signals.with_columns(
            [
                pl.col("spread")
                .rolling_mean(window_size=self.window, min_periods=1)
                .alias("spread_mean"),
                pl.col("spread")
                .rolling_std(window_size=self.window, min_periods=1)
                .alias("spread_std"),
            ]
        )

        # Calculate z-score, avoiding division by zero
        signals = signals.with_columns(
            [
                pl.when(pl.col("spread_std") != 0)
                .then((pl.col("spread") - pl.col("spread_mean")) / pl.col("spread_std"))
                .otherwise(0)
                .alias("z_score")
            ]
        )

        # Generate trading signals
        signals = signals.with_columns(
            [
                pl.when(pl.col("z_score") > self.entry_z_score)
                .then(-1)
                .when(pl.col("z_score") < -self.entry_z_score)
                .then(1)
                .when(pl.col("z_score").abs() < self.exit_z_score)
                .then(0)
                .otherwise(None)
                .alias("signal")
            ]
        )

        # Fill forward the signal
        signals = signals.with_columns(
            [pl.col("signal").forward_fill().fill_null(0).alias("signal")]
        )

        # Calculate positions (changes in signal)
        signals = signals.with_columns(
            [pl.col("signal").diff().fill_null(0).alias("positions")]
        )

        return signals


class BuyAndHoldStrategy(Strategy):
    """
    Implements a simple buy and hold strategy.
    """

    def __init__(self, params: dict[str, Any]):
        # No parameters needed for this strategy
        pass

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates a buy signal on the first day and holds.
        """
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("short_mavg", pl.Float64),
                    ("long_mavg", pl.Float64),
                    ("signal", pl.Float64),
                    ("positions", pl.Float64),
                ]
            )

        signals = data.select([pl.col("Date"), pl.col("Close")])
        signals = signals.with_columns(
            [
                pl.lit(1).alias("signal"),
                pl.lit(1).alias("positions").first().alias("positions"),
            ]
        )
        return signals
