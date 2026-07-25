"""
Implements the mean reversion strategy, which is based on the assumption
that asset prices tend to revert to their mean over time.
"""

from typing import Any

import math

import polars as pl

from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategy_params import (
    validate_strategy_params,
)


class MeanReversionStrategy(BaseStrategy):
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
        validate_strategy_params("Mean Reversion", params)
        super().__init__(params)
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
        band. Positions are held until the price crosses back through the
        rolling mean.

        Args:
            data: A DataFrame containing the price data. Must have a 'Close'
                  column.

        Returns:
            A DataFrame containing the generated trading signals. Columns
            include 'signal', 'mean', 'std', 'upper_band', 'lower_band', and
            'position_change'.
        """
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("mean", pl.Float64),
                    ("std", pl.Float64),
                    ("upper_band", pl.Float64),
                    ("lower_band", pl.Float64),
                    ("signal", pl.Float64),
                    ("position_change", pl.Float64),
                ]
            )

        indicators: pl.DataFrame = (  # type: ignore[invalid-assignment]
            data.select([pl.col("Date"), pl.col("Close")])
            .lazy()
            .with_columns(
                [
                    pl.col("Close")
                    .rolling_mean(
                        window_size=self.window,
                        min_samples=self.window,
                    )
                    .alias("mean"),
                    pl.col("Close")
                    .rolling_std(
                        window_size=self.window,
                        min_samples=self.window,
                    )
                    .alias("std"),
                ]
            )
            .with_columns(
                [
                    (pl.col("mean") + (self.std_dev * pl.col("std"))).alias(
                        "upper_band"
                    ),
                    (pl.col("mean") - (self.std_dev * pl.col("std"))).alias(
                        "lower_band"
                    ),
                ]
            )
            .collect()
        )
        signal_values = self._generate_stateful_signals(indicators)
        position_changes = self._calculate_position_changes(signal_values)

        return indicators.with_columns(
            [
                pl.Series("signal", signal_values, dtype=pl.Float64),
                pl.Series(
                    "position_change",
                    position_changes,
                    dtype=pl.Float64,
                ),
            ]
        )

    def _generate_stateful_signals(self, indicators: pl.DataFrame) -> list[float]:
        """Generate mean-reversion signals using entry and exit state."""
        signal = 0.0
        signals: list[float] = []

        for close, mean, std, upper_band, lower_band in indicators.select(
            ["Close", "mean", "std", "upper_band", "lower_band"]
        ).iter_rows():
            if not self._has_valid_band(close, mean, std, upper_band, lower_band):
                signal = 0.0
            else:
                close_value = float(close)
                mean_value = float(mean)
                upper_band_value = float(upper_band)
                lower_band_value = float(lower_band)

                if signal > 0:
                    signal = 0.0 if close_value >= mean_value else 1.0
                elif signal < 0:
                    signal = 0.0 if close_value <= mean_value else -1.0
                elif close_value < lower_band_value:
                    signal = 1.0
                elif close_value > upper_band_value:
                    signal = -1.0

            signals.append(signal)

        return signals

    @staticmethod
    def _has_valid_band(
        close: Any,
        mean: Any,
        std: Any,
        upper_band: Any,
        lower_band: Any,
    ) -> bool:
        """Return whether the row has a tradable rolling band."""
        return (
            MeanReversionStrategy._is_finite_number(close)
            and MeanReversionStrategy._is_finite_number(mean)
            and MeanReversionStrategy._is_finite_number(std)
            and MeanReversionStrategy._is_finite_number(upper_band)
            and MeanReversionStrategy._is_finite_number(lower_band)
            and float(std) > 0
        )

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """Return whether a value can be interpreted as a finite number."""
        if value is None:
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _calculate_position_changes(signals: list[float]) -> list[float]:
        """Calculate signal changes from a flat starting position."""
        previous_signal = 0.0
        position_changes = []
        for signal in signals:
            position_changes.append(signal - previous_signal)
            previous_signal = signal

        return position_changes
