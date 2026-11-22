"""Implement a pairs trading strategy for two correlated financial assets."""

import math
from typing import Any

import polars as pl

from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategy_params import (
    validate_strategy_params,
)

# Relative variance tolerance for guarding floating-point cancellation on
# collinear or constant-spread series.
_SPREAD_VARIANCE_REL_TOL = 1e-13


class PairsTradingStrategy(BaseStrategy):
    """Implement a market-neutral pairs trading strategy for two assets.

    This strategy estimates a positive rolling OLS hedge ratio, calculates the
    hedge-ratio-adjusted spread, and uses spread z-scores to enter, exit, or
    flip positions while locking leg weights at entry or flip.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        validate_strategy_params("Pairs Trading", params)
        super().__init__(params)
        # Lookback period for the rolling hedge ratio and spread statistics.
        self.window = int(params["window"])
        # Z-score threshold for entering or flipping a trade.
        self.entry_z_score = float(params["entry_z_score"])
        # Z-score threshold for exiting an open trade.
        self.exit_z_score = float(params["exit_z_score"])

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals based on the pairs trading strategy.

        Calculates the rolling OLS hedge ratio and spread between two assets,
        computes the window-consistent spread mean, sample standard deviation,
        and z-score using the current window's hedge ratio across the lookback
        window, and generates stateful signals and entry-locked leg weights
        using the entry and exit thresholds.

        Trading Logic:
            - Requires a finite positive rolling hedge ratio and positive
              spread standard deviation; invalid rows exit to flat.
            - Flat to short (-1) when z-score > entry_z_score; flat to long
              (1) when z-score < -entry_z_score.
            - Long flips to short (-1) when z-score > entry_z_score and exits
              to flat (0) when z-score >= -exit_z_score.
            - Short flips to long (1) when z-score < -entry_z_score and exits
              to flat (0) when z-score <= exit_z_score.
            - Leg weights are normalised by gross notional on entry or flip
              and held constant until the signal changes.

        Args:
            data: A DataFrame containing the price data. Must have 'Close_1'
                and 'Close_2' columns representing the closing prices of the
                two assets.

        Returns:
            A DataFrame containing the generated trading signals.
            Columns include:
            - 'hedge_ratio': The rolling OLS beta of asset 1 against asset 2.
            - 'spread': The hedge-ratio-adjusted spread between the two assets.
            - 'spread_mean': The lookback mean of the spread evaluated using
              the current window's 'hedge_ratio'.
            - 'spread_std': The lookback sample standard deviation (ddof=1) of
              the spread evaluated using the current window's 'hedge_ratio'.
            - 'z_score': The standardised score of the spread.
            - 'signal': The trading signal (-1, 0, or 1).
            - 'leg_1_weight' and 'leg_2_weight': The entry-locked normalised
              position weights for each leg.
            - 'leg_1_weight_change' and 'leg_2_weight_change': Period changes
              in each leg weight from a flat starting position.
            - 'position_change': The change in signal from the previous period.
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

        prices_are_valid = (
            pl.col("Close_1").is_not_null()
            & pl.col("Close_1").is_finite()
            & (pl.col("Close_1") > 0)
            & pl.col("Close_2").is_not_null()
            & pl.col("Close_2").is_finite()
            & (pl.col("Close_2") > 0)
        )
        hedge_ratio_is_valid = (
            pl.col("hedge_ratio").is_not_null()
            & pl.col("hedge_ratio").is_finite()
            & (pl.col("hedge_ratio") > 0)
        )
        valid_spread_band = (
            prices_are_valid
            & hedge_ratio_is_valid
            & pl.col("spread").is_not_null()
            & pl.col("spread").is_finite()
            & pl.col("spread_mean").is_not_null()
            & pl.col("spread_mean").is_finite()
            & pl.col("spread_std").is_not_null()
            & pl.col("spread_std").is_finite()
            & (pl.col("spread_std") > 0)
        )

        indicators: pl.DataFrame = (
            data.select(
                [
                    pl.col("Date"),
                    pl.col("Close_1"),
                    pl.col("Close_2"),
                ]
            )
            .lazy()
            # Estimate rolling moments (ddof=1) for OLS beta and spread stats.
            .with_columns(
                [
                    pl.col("Close_1")
                    .rolling_mean(window_size=self.window, min_samples=self.window)
                    .alias("close_1_mean"),
                    pl.col("Close_2")
                    .rolling_mean(window_size=self.window, min_samples=self.window)
                    .alias("close_2_mean"),
                    pl.col("Close_1")
                    .rolling_var(window_size=self.window, min_samples=self.window)
                    .alias("close_1_variance"),
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
                        & pl.col("close_covariance").is_not_null()
                        & pl.col("close_covariance").is_finite()
                    )
                    .then(pl.col("close_covariance") / pl.col("close_2_variance"))
                    .otherwise(None)
                    .alias("hedge_ratio")
                ]
            )
            # Evaluate spread and its lookback mean/variance with beta_t.
            .with_columns(
                [
                    (
                        pl.col("Close_1") - pl.col("hedge_ratio") * pl.col("Close_2")
                    ).alias("spread"),
                    pl.when(
                        pl.col("hedge_ratio").is_not_null()
                        & pl.col("hedge_ratio").is_finite()
                    )
                    .then(
                        pl.col("close_1_mean")
                        - pl.col("hedge_ratio") * pl.col("close_2_mean")
                    )
                    .otherwise(None)
                    .alias("spread_mean"),
                    pl.when(
                        pl.col("hedge_ratio").is_not_null()
                        & pl.col("hedge_ratio").is_finite()
                    )
                    .then(
                        pl.col("close_1_variance")
                        - 2.0 * pl.col("hedge_ratio") * pl.col("close_covariance")
                        + pl.col("hedge_ratio").pow(2) * pl.col("close_2_variance")
                    )
                    .otherwise(None)
                    .alias("spread_variance"),
                ]
            )
            # Clamp near-zero residual variance to 0.0 to avoid rounding noise.
            .with_columns(
                [
                    pl.when(
                        pl.col("spread_variance").is_null()
                        | ~pl.col("spread_variance").is_finite()
                        | pl.col("close_1_variance").is_null()
                        | ~pl.col("close_1_variance").is_finite()
                        | pl.col("close_1_mean").is_null()
                        | ~pl.col("close_1_mean").is_finite()
                        | pl.col("close_2_mean").is_null()
                        | ~pl.col("close_2_mean").is_finite()
                    )
                    .then(None)
                    .when(
                        pl.col("spread_variance")
                        <= _SPREAD_VARIANCE_REL_TOL
                        * (
                            pl.col("close_1_variance")
                            + pl.col("close_1_mean").pow(2)
                            + (
                                pl.col("hedge_ratio")
                                * pl.col("close_1_mean")
                                * pl.col("close_2_mean")
                            ).abs()
                        )
                    )
                    .then(0.0)
                    .otherwise(pl.col("spread_variance").sqrt())
                    .alias("spread_std")
                ]
            )
            # Calculate z-score, guarding zero std and non-positive beta.
            .with_columns(
                [
                    pl.when(valid_spread_band)
                    .then(
                        (pl.col("spread") - pl.col("spread_mean"))
                        / pl.col("spread_std")
                    )
                    .otherwise(0.0)
                    .alias("z_score")
                ]
            )
            .drop(
                [
                    "close_1_mean",
                    "close_2_mean",
                    "close_1_variance",
                    "close_2_variance",
                    "close_covariance",
                    "spread_variance",
                ]
            )
            .collect()
        )

        signal_values, leg_1_weights, leg_2_weights = self._generate_stateful_signals(
            indicators
        )
        position_changes = self._calculate_changes(signal_values)
        leg_1_weight_changes = self._calculate_changes(leg_1_weights)
        leg_2_weight_changes = self._calculate_changes(leg_2_weights)

        return indicators.with_columns(
            [
                pl.Series("signal", signal_values, dtype=pl.Float64),
                pl.Series("leg_1_weight", leg_1_weights, dtype=pl.Float64),
                pl.Series("leg_2_weight", leg_2_weights, dtype=pl.Float64),
                pl.Series(
                    "leg_1_weight_change",
                    leg_1_weight_changes,
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "leg_2_weight_change",
                    leg_2_weight_changes,
                    dtype=pl.Float64,
                ),
                pl.Series("position_change", position_changes, dtype=pl.Float64),
            ]
        )

    def _generate_stateful_signals(
        self, indicators: pl.DataFrame
    ) -> tuple[list[float], list[float], list[float]]:
        """Generate stateful pair signals and entry-locked leg weights."""
        signal = 0.0
        leg_1_weight = 0.0
        leg_2_weight = 0.0

        signals: list[float] = []
        leg_1_weights: list[float] = []
        leg_2_weights: list[float] = []

        for (
            close_1,
            close_2,
            hedge_ratio,
            spread,
            spread_mean,
            spread_std,
            z_score,
        ) in indicators.select(
            [
                "Close_1",
                "Close_2",
                "hedge_ratio",
                "spread",
                "spread_mean",
                "spread_std",
                "z_score",
            ]
        ).iter_rows():
            if not self._has_valid_row(
                close_1,
                close_2,
                hedge_ratio,
                spread,
                spread_mean,
                spread_std,
                z_score,
            ):
                signal = 0.0
                leg_1_weight = 0.0
                leg_2_weight = 0.0
            else:
                close_1_val = float(close_1)
                close_2_val = float(close_2)
                hedge_ratio_val = float(hedge_ratio)
                z_score_val = float(z_score)

                leg_1_notional = abs(close_1_val)
                leg_2_notional = abs(hedge_ratio_val * close_2_val)
                gross_notional = leg_1_notional + leg_2_notional

                if not math.isfinite(gross_notional) or gross_notional <= 0.0:
                    signal = 0.0
                    leg_1_weight = 0.0
                    leg_2_weight = 0.0
                else:
                    new_signal = self._next_signal(signal, z_score_val)
                    if new_signal == 0.0:
                        leg_1_weight = 0.0
                        leg_2_weight = 0.0
                    elif new_signal != signal:
                        leg_1_weight = new_signal * leg_1_notional / gross_notional
                        leg_2_weight = -new_signal * leg_2_notional / gross_notional
                    signal = new_signal

            signals.append(signal)
            leg_1_weights.append(leg_1_weight)
            leg_2_weights.append(leg_2_weight)

        return signals, leg_1_weights, leg_2_weights

    def _next_signal(self, current_signal: float, z_score: float) -> float:
        """Return the next signal state from the current signal and z-score."""
        if current_signal > 0.0:
            if z_score > self.entry_z_score:
                return -1.0
            if z_score >= -self.exit_z_score:
                return 0.0
            return 1.0

        if current_signal < 0.0:
            if z_score < -self.entry_z_score:
                return 1.0
            if z_score <= self.exit_z_score:
                return 0.0
            return -1.0

        if z_score > self.entry_z_score:
            return -1.0
        if z_score < -self.entry_z_score:
            return 1.0
        return 0.0

    @staticmethod
    def _has_valid_row(
        close_1: Any,
        close_2: Any,
        hedge_ratio: Any,
        spread: Any,
        spread_mean: Any,
        spread_std: Any,
        z_score: Any,
    ) -> bool:
        """Return whether a row has valid prices, positive beta, and spread."""
        return (
            PairsTradingStrategy._is_finite_number(close_1)
            and float(close_1) > 0.0
            and PairsTradingStrategy._is_finite_number(close_2)
            and float(close_2) > 0.0
            and PairsTradingStrategy._is_finite_number(hedge_ratio)
            and float(hedge_ratio) > 0.0
            and PairsTradingStrategy._is_finite_number(spread)
            and PairsTradingStrategy._is_finite_number(spread_mean)
            and PairsTradingStrategy._is_finite_number(spread_std)
            and float(spread_std) > 0.0
            and PairsTradingStrategy._is_finite_number(z_score)
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
    def _calculate_changes(values: list[float]) -> list[float]:
        """Calculate step changes from a flat starting value."""
        previous_value = 0.0
        changes: list[float] = []
        for value in values:
            changes.append(value - previous_value)
            previous_value = value

        return changes
