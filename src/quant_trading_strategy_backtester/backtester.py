"""
A backtesting framework for quantitative trading strategies.

It provides a Backtester class that can run a given trading strategy on
historical data and calculate various performance metrics. This framework is
designed to work with the strategy templates defined in a different module in
this repository.
"""

import json
import platform
from datetime import date, datetime
from typing import Any

import polars as pl
import streamlit as st

from quant_trading_strategy_backtester.models import Session
from quant_trading_strategy_backtester.models import StrategyModel as StrategyModel
from quant_trading_strategy_backtester.strategies.base import BaseStrategy

TRADING_DAYS_PER_YEAR = 252

TRADE_LEDGER_SCHEMA = {
    "Date": pl.Date,
    "Action": pl.Utf8,
    "Reason": pl.Utf8,
    "Signal": pl.Float64,
    "Previous Signal": pl.Float64,
    "Position Change": pl.Float64,
    "Turnover": pl.Float64,
    "Gross Return": pl.Float64,
    "Transaction Costs": pl.Float64,
    "Net Return": pl.Float64,
    "Cumulative Costs": pl.Float64,
    "Equity": pl.Float64,
    "Holding Period Days": pl.Int64,
    "Leg 1 Weight": pl.Float64,
    "Leg 2 Weight": pl.Float64,
    "Z-Score": pl.Float64,
}


def is_running_locally() -> bool:
    """
    Determines if the app is running locally or on Streamlit Cloud by checking
    platform characteristics. platform.processor() returns nothing on Streamlit
    Cloud but returns a value locally.

    Returns:
        bool: True if running locally, False if running on Streamlit Cloud
    """
    return bool(platform.processor())


def build_trade_ledger(results: pl.DataFrame) -> pl.DataFrame:
    """
    Build a row-level trade ledger from backtest results.

    Args:
        results: The backtest results DataFrame.

    Returns:
        A DataFrame containing rows where position or leg exposure changed.
    """
    if results.is_empty():
        return pl.DataFrame(schema=TRADE_LEDGER_SCHEMA)

    required_columns = {
        "Date",
        "signal",
        "position_change",
        "trade_turnover",
        "gross_strategy_returns",
        "transaction_costs",
        "strategy_returns",
        "cumulative_transaction_costs",
        "equity_curve",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            f"Trade ledger requires backtest result columns: {sorted(missing_columns)}"
        )

    records: list[dict[str, Any]] = []
    position_start_date: date | datetime | None = None
    previous_signal = 0.0

    for row in results.iter_rows(named=True):
        signal = float(row["signal"] or 0.0)
        position_change = float(row["position_change"] or 0.0)
        turnover = float(row["trade_turnover"] or 0.0)

        if turnover <= 0 and abs(position_change) <= 0:
            previous_signal = signal
            continue

        current_date = _normalise_trade_date(row["Date"])
        action = _classify_trade_action(previous_signal, signal, turnover)
        holding_period_days = _calculate_holding_period_days(
            current_date, position_start_date
        )

        records.append(
            {
                "Date": current_date,
                "Action": action,
                "Reason": _classify_trade_reason(row, action),
                "Signal": signal,
                "Previous Signal": previous_signal,
                "Position Change": position_change,
                "Turnover": turnover,
                "Gross Return": float(row["gross_strategy_returns"] or 0.0),
                "Transaction Costs": float(row["transaction_costs"] or 0.0),
                "Net Return": float(row["strategy_returns"] or 0.0),
                "Cumulative Costs": float(row["cumulative_transaction_costs"] or 0.0),
                "Equity": float(row["equity_curve"] or 0.0),
                "Holding Period Days": holding_period_days,
                "Leg 1 Weight": _optional_float(row, "leg_1_weight"),
                "Leg 2 Weight": _optional_float(row, "leg_2_weight"),
                "Z-Score": _optional_float(row, "z_score"),
            }
        )

        if previous_signal == 0 and signal != 0:
            position_start_date = current_date
        elif previous_signal != 0 and signal == 0:
            position_start_date = None
        elif previous_signal * signal < 0:
            position_start_date = current_date

        previous_signal = signal

    return pl.DataFrame(records, schema=TRADE_LEDGER_SCHEMA)


def _classify_trade_action(
    previous_signal: float, signal: float, turnover: float
) -> str:
    """Classify a trade event from previous and current exposure."""
    if previous_signal == 0 and signal > 0:
        return "Enter Long"
    if previous_signal == 0 and signal < 0:
        return "Enter Short"
    if previous_signal > 0 and signal == 0:
        return "Exit Long"
    if previous_signal < 0 and signal == 0:
        return "Exit Short"
    if previous_signal > 0 and signal < 0:
        return "Flip Long To Short"
    if previous_signal < 0 and signal > 0:
        return "Flip Short To Long"
    if turnover > 0 and signal != 0:
        return "Rebalance"

    return "Trade"


def _classify_trade_reason(row: dict[str, Any], action: str) -> str:
    """Return the signal reason for a trade ledger row."""
    if action == "Rebalance":
        return "Leg weights changed"
    if row.get("z_score") is not None:
        return "Pairs z-score signal"

    return "Strategy signal changed"


def _calculate_holding_period_days(
    current_date: date | datetime,
    position_start_date: date | datetime | None,
) -> int | None:
    """Return position age in calendar days for open or closing trades."""
    if position_start_date is None:
        return None

    return (current_date - position_start_date).days


def _normalise_trade_date(value: date | datetime) -> date:
    """Return a plain date for ledger display and schema construction."""
    if isinstance(value, datetime):
        return value.date()

    return value


def _optional_float(row: dict[str, Any], name: str) -> float | None:
    """Return an optional row value as a float."""
    value = row.get(name)
    if value is None:
        return None

    return float(value)


class Backtester:
    """
    Backtests trading strategies.

    Takes historical data and a trading strategy, runs the strategy on the
    data, and calculates performance metrics.

    Attributes:
        data: Historical price data.
        strategy: The trading strategy to backtest.
        initial_capital: The initial capital for the backtest.
        transaction_cost_bps: Transaction cost per trade in basis points.
        slippage_bps: Slippage per trade in basis points.
        results: The results of the backtest (initialised after running).
        tickers: The ticker or tickers used in the backtest.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 3.0,
        session=None,
        tickers: str | list[str] | None = None,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.results: None | pl.DataFrame = None
        self.session = session or Session()
        self.tickers = tickers

    def run(self) -> pl.DataFrame:
        """
        Runs the backtest.

        Generates trading signals using the strategy, calculates returns,
        and stores the results. Call save_results() separately to persist
        to the database.

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

        if "Close_1" in self.data.columns and "Close_2" in self.data.columns:
            return self._calculate_pairs_returns(signals)

        if "Close" in self.data.columns:
            return self._calculate_single_asset_returns(signals)

        raise ValueError("Data does not contain required 'Close' columns")

    def _calculate_single_asset_returns(self, signals: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates returns for single-asset strategies.

        Args:
            signals: The trading signals generated by the strategy.

        Returns:
            A DataFrame containing calculated returns and related metrics.
        """
        asset_returns = (self.data["Close"] - self.data["Close"].shift(1)) / self.data[
            "Close"
        ].shift(1)
        cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10_000

        portfolio = (
            signals.lazy()
            .with_columns(
                [
                    pl.col("position_change"),
                    asset_returns.alias("asset_returns"),
                    (pl.col("signal").shift(1) * asset_returns).alias(
                        "gross_strategy_returns"
                    ),
                ]
            )
            # Handle potential NaN or inf values.
            .with_columns(
                [
                    pl.col("gross_strategy_returns")
                    .replace({float("inf"): None, float("-inf"): None})
                    .fill_null(0)
                    .alias("gross_strategy_returns"),
                    pl.col("position_change").abs().alias("trade_turnover"),
                ]
            )
            # Deduct transaction costs and slippage on position changes.
            .with_columns(
                [
                    (pl.col("trade_turnover") * cost_rate).alias("transaction_costs"),
                    (
                        pl.col("gross_strategy_returns")
                        - pl.col("trade_turnover") * cost_rate
                    ).alias("strategy_returns"),
                ]
            )
        )

        return self._add_cumulative_returns(portfolio)

    def _calculate_pairs_returns(self, signals: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates returns for hedge-ratio-weighted pairs strategies.

        Args:
            signals: The trading signals generated by the strategy.

        Returns:
            A DataFrame containing calculated returns and related metrics.

        Raises:
            ValueError: If the signals do not contain pair leg weights.
        """
        required_columns = {
            "leg_1_weight",
            "leg_2_weight",
            "leg_1_weight_change",
            "leg_2_weight_change",
        }
        missing_columns = required_columns - set(signals.columns)
        if missing_columns:
            raise ValueError(
                "Pairs trading signals must contain leg weight columns: "
                f"{sorted(missing_columns)}"
            )

        asset_1_returns = (
            self.data["Close_1"] - self.data["Close_1"].shift(1)
        ) / self.data["Close_1"].shift(1)
        asset_2_returns = (
            self.data["Close_2"] - self.data["Close_2"].shift(1)
        ) / self.data["Close_2"].shift(1)
        cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10_000

        portfolio = (
            signals.lazy()
            .with_columns(
                [
                    asset_1_returns.alias("asset_1_returns"),
                    asset_2_returns.alias("asset_2_returns"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("leg_1_weight") * pl.col("asset_1_returns")
                        + pl.col("leg_2_weight") * pl.col("asset_2_returns")
                    ).alias("asset_returns"),
                    (
                        pl.col("leg_1_weight").shift(1).fill_null(0)
                        * pl.col("asset_1_returns")
                        + pl.col("leg_2_weight").shift(1).fill_null(0)
                        * pl.col("asset_2_returns")
                    ).alias("gross_strategy_returns"),
                    (
                        pl.col("leg_1_weight_change").abs()
                        + pl.col("leg_2_weight_change").abs()
                    ).alias("trade_turnover"),
                ]
            )
            # Handle potential NaN or inf values.
            .with_columns(
                [
                    pl.col("gross_strategy_returns")
                    .replace({float("inf"): None, float("-inf"): None})
                    .fill_null(0)
                    .alias("gross_strategy_returns"),
                    pl.col("trade_turnover").alias("pair_turnover"),
                ]
            )
            # Deduct transaction costs and slippage on leg turnover.
            .with_columns(
                [
                    (pl.col("trade_turnover") * cost_rate).alias("transaction_costs"),
                    (
                        pl.col("gross_strategy_returns")
                        - pl.col("trade_turnover") * cost_rate
                    ).alias("strategy_returns"),
                ]
            )
        )

        return self._add_cumulative_returns(portfolio)

    def _add_cumulative_returns(self, portfolio: pl.LazyFrame) -> pl.DataFrame:
        """Add cumulative returns and equity curve columns."""
        result = portfolio.with_columns(
            [
                (1 + pl.col("gross_strategy_returns"))
                .cum_prod()
                .alias("gross_cumulative_returns"),
                (1 + pl.col("strategy_returns")).cum_prod().alias("cumulative_returns"),
                (
                    self.initial_capital * (1 + pl.col("strategy_returns")).cum_prod()
                ).alias("equity_curve"),
                pl.col("transaction_costs")
                .cum_sum()
                .alias("cumulative_transaction_costs"),
            ]
        ).collect()
        if not isinstance(result, pl.DataFrame):
            raise TypeError(
                "Expected cumulative return calculation to collect a DataFrame"
            )

        return result

    def get_performance_metrics(
        self, risk_free_return_rate_annual: float = 0.0
    ) -> dict[str, float] | None:
        """
        Calculates key performance metrics from the trading strategy backtest.

        Computes the total return, Sharpe ratio, and maximum drawdown based on
        the backtest results.

        Args:
            - risk_free_return_rate_annual: The risk-free rate of return
                annualised. Defaults to 0.0% per annum for simplicity.

        Returns:
            A dictionary containing performance metrics, or None if the
            backtest hasn't been run yet.
        """
        if self.results is None:
            return None

        total_return = (
            float(self.results["cumulative_returns"].cast(pl.Float64).tail(1).item())
            - 1
        )
        gross_total_return = (
            float(
                self.results["gross_cumulative_returns"].cast(pl.Float64).tail(1).item()
            )
            - 1
        )
        total_costs = float(self.results["transaction_costs"].cast(pl.Float64).sum())
        total_turnover = float(self.results["trade_turnover"].cast(pl.Float64).sum())
        trade_events = float(self.results.filter(pl.col("trade_turnover") > 0).height)

        # Measure the risk-adjusted return.
        periods = TRADING_DAYS_PER_YEAR
        rf_daily = (1 + risk_free_return_rate_annual) ** (1 / periods) - 1
        excess = self.results["strategy_returns"].cast(pl.Float64) - rf_daily
        returns_mean = float(excess.mean())
        returns_std = float(excess.std())
        sharpe_ratio = (
            float((periods**0.5) * returns_mean / returns_std)
            if returns_std
            else float("nan")
        )

        # Measure the maximum loss from a peak to a trough of the equity curve.
        drawdowns = (
            self.results["equity_curve"] / self.results["equity_curve"].cum_max() - 1
        )
        max_drawdown = float(drawdowns.cast(pl.Float64).min())  # type: ignore

        # TODO: Split the calculations out into separate functions.
        return {
            "Total Return": total_return,
            "Gross Total Return": gross_total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Total Costs": total_costs,
            "Cost Drag": gross_total_return - total_return,
            "Trade Events": trade_events,
            "Total Turnover": total_turnover,
        }

    def save_results(self) -> None:
        """
        Saves the strategy and its backtest results to either the local database
        or session state, depending on the environment.
        """
        metrics = self.get_performance_metrics()
        if metrics is None:
            raise ValueError("Backtest hasn't been run yet. Call run() first.")

        strategy_params = self.strategy.get_parameters()
        strategy_name = self.strategy.__class__.__name__

        # Determine start and end dates
        start_date_row = self.data.select(
            pl.col("Date").dt.year().alias("year"),
            pl.col("Date").dt.month().alias("month"),
            pl.col("Date").dt.day().alias("day"),
        ).row(0)
        end_date_row = self.data.select(
            pl.col("Date").dt.year().alias("year"),
            pl.col("Date").dt.month().alias("month"),
            pl.col("Date").dt.day().alias("day"),
        ).row(-1)

        start_date = date(start_date_row[0], start_date_row[1], start_date_row[2])
        end_date = date(end_date_row[0], end_date_row[1], end_date_row[2])

        if is_running_locally():
            try:
                # Check if a strategy with the same name, parameters, and date
                # range already exists
                existing_strategy = (
                    self.session.query(StrategyModel)
                    .filter_by(
                        name=strategy_name,
                        parameters=json.dumps(strategy_params),
                        start_date=start_date,
                        end_date=end_date,
                    )
                    .first()
                )

                if existing_strategy is None:
                    new_strategy = StrategyModel(
                        name=strategy_name,
                        parameters=json.dumps(strategy_params),
                        total_return=metrics["Total Return"],
                        sharpe_ratio=metrics["Sharpe Ratio"],
                        max_drawdown=metrics["Max Drawdown"],
                        tickers=json.dumps(self.tickers),
                        start_date=start_date,
                        end_date=end_date,
                    )
                    self.session.add(new_strategy)
                    self.session.commit()
                    print(f"Strategy {strategy_name} saved successfully.")
                else:
                    print(
                        f"Strategy {strategy_name} with same parameters already exists. Skipping save."
                    )
            except Exception as e:
                self.session.rollback()
                raise ValueError(f"Failed to save strategy results: {str(e)}")
        else:
            # Use Streamlit session state for cloud deployment
            if "strategy_results" not in st.session_state:
                st.session_state.strategy_results = []

            st.session_state.strategy_results.append(
                {
                    "date_created": datetime.now(),
                    "name": strategy_name,
                    "parameters": strategy_params,
                    "total_return": metrics["Total Return"],
                    "sharpe_ratio": metrics["Sharpe Ratio"],
                    "max_drawdown": metrics["Max Drawdown"],
                    "tickers": self.tickers,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            print(f"Strategy {strategy_name} saved to session state.")
