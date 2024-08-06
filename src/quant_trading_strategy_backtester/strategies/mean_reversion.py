"""
Implements the mean reversion strategy, which is based on the assumptio that
asset prices tend to revert to their mean over time.
"""
from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import Strategy


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
