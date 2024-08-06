"""
Implements a pairs trading strategy for two correlated financial instruments,
which is based on the assumption that the historical relationship between two
assets will continue.
"""

from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import Strategy


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
            - 'z_score': The standardised score of the spread.
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
