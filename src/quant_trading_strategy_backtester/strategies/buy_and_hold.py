from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import Strategy


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
