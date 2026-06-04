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
        """Generate trading signals for the Buy and Hold strategy.

        Create a constant long position with a single position change on the
        first row, so costs are charged once when the strategy enters.

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
                    ("position_change", pl.Float64),
                ]
            )

        signals: pl.DataFrame = (  # type: ignore[invalid-assignment]
            data.select([pl.col("Date"), pl.col("Close")])
            .with_row_index()
            .lazy()
            .with_columns(
                [
                    pl.lit(1.0).alias("signal"),
                    pl.when(pl.col("index") == 0)
                    .then(1.0)
                    .otherwise(0.0)
                    .alias("position_change"),
                ]
            )
            .drop("index")
            .collect()
        )

        return signals
