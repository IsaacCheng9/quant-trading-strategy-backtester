import numpy as np
import pandas as pd
from quant_trading_strategy_backtester.strategy_templates import Strategy


class Backtester:
    def __init__(
        self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000.0
    ):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None

    def run(self):
        signals = self.strategy.generate_signals(self.data)
        self.results = self._calculate_returns(signals)
        return self.results

    def _calculate_returns(self, signals: pd.DataFrame) -> pd.DataFrame:
        portfolio = pd.DataFrame(index=signals.index)
        portfolio["positions"] = signals["positions"]
        portfolio["asset_returns"] = self.data["Close"].pct_change()
        portfolio["strategy_returns"] = (
            portfolio["positions"].shift(1) * portfolio["asset_returns"]
        )
        portfolio["cumulative_returns"] = (1 + portfolio["strategy_returns"]).cumprod()
        portfolio["equity_curve"] = (
            self.initial_capital * portfolio["cumulative_returns"]
        )

        return portfolio

    def get_performance_metrics(self) -> dict[str, float] | None:
        if self.results is None:
            return None

        total_return = self.results["cumulative_returns"].iloc[-1] - 1
        # Measure the risk-adjusted return, assuming 252 trading days per year.
        sharpe_ratio = (
            np.sqrt(252)
            * self.results["strategy_returns"].mean()
            / self.results["strategy_returns"].std()
        )
        # Measure the maximum loss from a peak to a trough of the equity curve.
        max_drawdown = (
            self.results["equity_curve"] / self.results["equity_curve"].cummax() - 1
        ).min()

        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
        }
