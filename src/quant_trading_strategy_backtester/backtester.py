"""
A backtesting framework for quantitative trading strategies.

It provides a Backtester class that can run a given trading strategy on
historical data and calculate various performance metrics. This framework is
designed to work with the strategy templates defined in a different module in
this repository.
"""

import polars as pl
from quant_trading_strategy_backtester.strategy_templates import Strategy


class Backtester:
    """
    Backtests trading strategies.

    Takes historical data and a trading strategy, runs the strategy on the
    data, and calculates performance metrics.

    Attributes:
        data: Historical price data.
        strategy: The trading strategy to backtest.
        initial_capital: The initial capital for the backtest.
        results: The results of the backtest (initialised after running).
    """

    def __init__(
        self, data: pl.DataFrame, strategy: Strategy, initial_capital: float = 100000.0
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None

    def run(self) -> pl.DataFrame:
        """
        Runs the backtest.

        Generates trading signals using the strategy, calculates returns,
        and stores the results.

        Returns:
            A DataFrame containing the backtest results.
        """
        signals = self.strategy.generate_signals(self.data)
        self.results = self._calculate_returns(signals)
        return self.results

    def _calculate_returns(self, signals: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates returns based on the generated signals.

        Computes asset returns, strategy returns, cumulative returns, and the
        equity curve.

        Args:
            signals: The trading signals generated by the strategy.

        Returns:
            A DataFrame containing calculated returns and related metrics.
        """
        # Ensure 'Date' column is present in signals DataFrame
        if "Date" not in signals.columns:
            raise ValueError("'Date' column is missing from the signals DataFrame")

        # Pairs trading
        if "Close_1" in self.data.columns and "Close_2" in self.data.columns:
            asset_returns = (
                self.data["Close_1"] - self.data["Close_1"].shift(1)
            ) / self.data["Close_1"].shift(1) - (
                self.data["Close_2"] - self.data["Close_2"].shift(1)
            ) / self.data["Close_2"].shift(1)
        # Single asset trading
        elif "Close" in self.data.columns:
            asset_returns = (
                self.data["Close"] - self.data["Close"].shift(1)
            ) / self.data["Close"].shift(1)
        else:
            raise ValueError("Data does not contain required 'Close' columns")

        portfolio = signals.with_columns(
            [
                pl.col("positions"),
                asset_returns.alias("asset_returns"),
                (pl.col("positions").shift(1) * asset_returns).alias(
                    "strategy_returns"
                ),
            ]
        )

        # Handle potential NaN or inf values
        portfolio = portfolio.with_columns(
            [
                pl.col("strategy_returns")
                .replace({float("inf"): None, float("-inf"): None})
                .fill_null(0)
            ]
        )

        portfolio = portfolio.with_columns(
            [
                (1 + pl.col("strategy_returns")).cum_prod().alias("cumulative_returns"),
                (
                    self.initial_capital * (1 + pl.col("strategy_returns")).cum_prod()
                ).alias("equity_curve"),
            ]
        )

        return portfolio

    def get_performance_metrics(self) -> dict[str, float] | None:
        """
        Calculates key performance metrics from the trading strategy backtest.

        Computes the total return, Sharpe ratio, and maximum drawdown based on
        the backtest results.

        Returns:
            A dictionary containing performance metrics, or None if the backtest
            hasn't been run yet.
        """
        if self.results is None:
            return None

        total_return = self.results["cumulative_returns"].tail(1)[0] - 1

        # Measure the risk-adjusted return, assuming 252 trading days per year.
        returns_mean = self.results["strategy_returns"].mean()
        returns_std = self.results["strategy_returns"].std()
        if returns_std != 0 and not pl.Series([returns_std]).is_nan()[0]:
            sharpe_ratio = (252**0.5) * returns_mean / returns_std
        else:
            sharpe_ratio = float("nan")

        # Measure the maximum loss from a peak to a trough of the equity curve.
        drawdowns = (
            self.results["equity_curve"] / self.results["equity_curve"].cum_max() - 1
        )
        max_drawdown = drawdowns.min()

        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
        }
