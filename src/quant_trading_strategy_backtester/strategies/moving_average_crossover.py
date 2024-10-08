"""
Implements the moving average crossover strategy, which is based on the
crossover of short-term and long-term moving averages of the closing price.
"""

from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import Strategy


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
        super().__init__(params)
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
