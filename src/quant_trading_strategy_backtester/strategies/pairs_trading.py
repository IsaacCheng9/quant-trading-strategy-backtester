"""
Implements a pairs trading strategy for two correlated financial instruments,
which is based on the assumption that the historical relationship between two
assets will continue.
"""

from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategy_params import (
    validate_strategy_params,
)


class PairsTradingStrategy(BaseStrategy):
    """
    Implements a pairs trading strategy for two correlated financial
    instruments, which is based on the assumption that the historical
    relationship between two assets will continue. This is a market-neutral
    strategy that exploits temporary mispricings in this relationship. It
    estimates a rolling hedge ratio, calculates a hedge-ratio-adjusted spread,
    and uses z-scores to determine when to enter and exit positions.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        validate_strategy_params("Pairs Trading", params)
        super().__init__(params)
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

        Calculates the rolling hedge ratio and spread between two assets,
        computes the z-score of the spread, and generates trading signals based
        on the entry and exit z-score thresholds.

        Trading Logic:
            - When z-score > entry_z_score:
                - Short asset 1, long the hedge-ratio-weighted asset 2
                  (signal = -1)
            - When z-score < -entry_z_score:
                - Long asset 1, short the hedge-ratio-weighted asset 2
                  (signal = 1)
            - When |z-score| < exit_z_score:
                - Close positions (signal = 0)

        Args:
            data: A DataFrame containing the price data. Must have 'Close_1'
                  and 'Close_2' columns representing the closing prices of the
                  two assets.

        Returns:
            A DataFrame containing the generated trading signals.
            Columns include:
            - 'hedge_ratio': The rolling OLS beta of asset 1 against asset 2.
            - 'spread': The hedge-ratio-adjusted spread between the two assets.
            - 'z_score': The standardised score of the spread.
            - 'signal': The trading signal (-1, 0, or 1).
            - 'leg_1_weight' and 'leg_2_weight': The normalised position
              weights for each leg.
            - 'position_change': The change in position from the previous period.
        """
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close_1", pl.Float64),
                    ("Close_2", pl.Float64),
                    ("hedge_ratio", pl.Float64),
                    ("spread", pl.Float64),
                    ("spread_mean", pl.Float64),
                    ("spread_std", pl.Float64),
                    ("z_score", pl.Float64),
                    ("signal", pl.Float64),
                    ("leg_1_weight", pl.Float64),
                    ("leg_2_weight", pl.Float64),
                    ("leg_1_weight_change", pl.Float64),
                    ("leg_2_weight_change", pl.Float64),
                    ("position_change", pl.Float64),
                ]
            )
        if "Close_1" not in data.columns or "Close_2" not in data.columns:
            raise ValueError("Data must contain 'Close_1' and 'Close_2' columns")

        hedge_ratio_is_valid = (
            pl.col("hedge_ratio").is_not_null() & pl.col("hedge_ratio").is_finite()
        )
        valid_spread_band = (
            hedge_ratio_is_valid
            & pl.col("spread_std").is_not_null()
            & pl.col("spread_std").is_finite()
            & (pl.col("spread_std") > 0)
        )

        signals: pl.DataFrame = (  # type: ignore[invalid-assignment]
            data.select(
                [
                    pl.col("Date"),
                    pl.col("Close_1"),
                    pl.col("Close_2"),
                ]
            )
            .lazy()
            # Estimate the rolling OLS beta of asset 1 against asset 2.
            .with_columns(
                [
                    pl.col("Close_2")
                    .rolling_var(window_size=self.window, min_samples=self.window)
                    .alias("close_2_variance"),
                    pl.rolling_cov(
                        "Close_1",
                        "Close_2",
                        window_size=self.window,
                        min_samples=self.window,
                    ).alias("close_covariance"),
                ]
            )
            .with_columns(
                [
                    pl.when(
                        pl.col("close_2_variance").is_not_null()
                        & pl.col("close_2_variance").is_finite()
                        & (pl.col("close_2_variance") > 0)
                    )
                    .then(pl.col("close_covariance") / pl.col("close_2_variance"))
                    .otherwise(None)
                    .alias("hedge_ratio")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("Close_1") - pl.col("hedge_ratio") * pl.col("Close_2")
                    ).alias("spread")
                ]
            )
            # Calculate rolling mean and std.
            .with_columns(
                [
                    pl.col("spread")
                    .rolling_mean(
                        window_size=self.window,
                        min_samples=self.window,
                    )
                    .alias("spread_mean"),
                    pl.col("spread")
                    .rolling_std(
                        window_size=self.window,
                        min_samples=self.window,
                    )
                    .alias("spread_std"),
                ]
            )
            # Calculate z-score, avoiding division by zero.
            .with_columns(
                [
                    pl.when(valid_spread_band)
                    .then(
                        (pl.col("spread") - pl.col("spread_mean"))
                        / pl.col("spread_std")
                    )
                    .otherwise(0)
                    .alias("z_score")
                ]
            )
            # Generate trading signals.
            .with_columns(
                [
                    pl.when(~valid_spread_band)
                    .then(0)
                    .when(valid_spread_band & (pl.col("z_score") > self.entry_z_score))
                    .then(-1)
                    .when(valid_spread_band & (pl.col("z_score") < -self.entry_z_score))
                    .then(1)
                    .when(
                        valid_spread_band
                        & (pl.col("z_score").abs() < self.exit_z_score)
                    )
                    .then(0)
                    .otherwise(None)
                    .alias("signal")
                ]
            )
            # Fill forward the signal.
            .with_columns(
                [pl.col("signal").forward_fill().fill_null(0).alias("signal")]
            )
            # Convert spread direction into normalised long/short leg weights.
            .with_columns(
                [
                    (1.0 + pl.col("hedge_ratio").abs()).alias("gross_exposure"),
                ]
            )
            .with_columns(
                [
                    pl.when(hedge_ratio_is_valid & (pl.col("signal") != 0))
                    .then(pl.col("signal") / pl.col("gross_exposure"))
                    .otherwise(0.0)
                    .alias("leg_1_weight"),
                    pl.when(hedge_ratio_is_valid & (pl.col("signal") != 0))
                    .then(
                        -pl.col("signal")
                        * pl.col("hedge_ratio")
                        / pl.col("gross_exposure")
                    )
                    .otherwise(0.0)
                    .alias("leg_2_weight"),
                ]
            )
            # Calculate position and leg-weight changes.
            .with_columns(
                [
                    pl.col("signal").diff().fill_null(0).alias("position_change"),
                    pl.col("leg_1_weight")
                    .diff()
                    .fill_null(pl.col("leg_1_weight"))
                    .alias("leg_1_weight_change"),
                    pl.col("leg_2_weight")
                    .diff()
                    .fill_null(pl.col("leg_2_weight"))
                    .alias("leg_2_weight_change"),
                ]
            )
            .drop(["close_2_variance", "close_covariance", "gross_exposure"])
            .with_columns(
                [
                    pl.col("signal").cast(pl.Float64),
                    pl.col("position_change").cast(pl.Float64),
                ]
            )
            .collect()
        )

        return signals
