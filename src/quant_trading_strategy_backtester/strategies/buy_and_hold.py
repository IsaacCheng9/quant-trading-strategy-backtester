"""
Implements the buy and hold strategy, which generates a buy signal on the first
day and holds the position indefinitely. This is a simple strategy that serves
as a benchmark to compare other trading strategies against.
"""

from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Implements a simple buy and hold strategy, which generates a buy signal on
    the first day and holds the position indefinitely.
    """

    def __init__(self, params: dict[str, Any]):
        # No additional parameters needed for this strategy, as it's always the same.
        super().__init__(params)

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates trading signals for the Buy and Hold strategy. Creates a
        column of 1s to indicate a constant long position.

        Args:
            data: Historical price data.

        Returns:
            A Polars DataFrame containing the trading signals.
        """
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("signal", pl.Float64),
                    ("positions", pl.Float64),
                ]
            )

        signals = data.select([pl.col("Date"), pl.col("Close")])
        # Add the 'signal' and 'positions' columns.
        signals = signals.with_columns(
            [
                pl.lit(1.0).alias("signal"),
                pl.lit(1.0).alias("positions"),
            ]
        )

        return signals
