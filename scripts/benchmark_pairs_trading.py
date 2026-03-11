"""
Benchmark pairs trading with parameter optimisation.

Pre-downloads data for all ticker pairs, then times only the computation
(signal generation + return calculation + metrics) over all pairs and parameter
combinations. This isolates the Polars computation from network I/O.
"""

import csv
import datetime
import itertools
import time
from typing import cast

import pandas as pd
import polars as pl
import yfinance as yf

from quant_trading_strategy_backtester.backtester import Backtester
from quant_trading_strategy_backtester.strategies.pairs_trading import (
    PairsTradingStrategy,
)

# Pinned top 20 S&P 500 companies by market cap for consistent
# benchmarks across runs.
TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "TSLA",
    "V",
    "UNH",
    "WMT",
    "XOM",
    "MA",
    "PG",
    "COST",
    "HD",
    "JNJ",
]
START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2023, 12, 31)
PARAM_RANGES = {
    "window": list(range(10, 101, 10)),
    "entry_z_score": [1.0, 1.5, 2.0, 2.5, 3.0],
    "exit_z_score": [0.1, 0.5, 1.0, 1.5],
}
NUM_RUNS = 6
OUTPUT_CSV = "resources/polars_pairs_trading_benchmark.csv"


def download_pair_data(ticker1: str, ticker2: str) -> pl.DataFrame | None:
    """
    Download and format price data for a ticker pair.

    Returns:
        A Polars DataFrame with Date, Close_1, Close_2 columns,
        or None if the download fails.
    """
    try:
        raw = yf.download(
            [ticker1, ticker2],
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
        )
        raw = cast(pd.DataFrame, raw)
        if not isinstance(raw.columns, pd.MultiIndex):
            return None

        close_data = raw["Close"].reset_index()
        close_data.columns = ["Date", "Close_1", "Close_2"]
        result = pl.from_pandas(close_data).drop_nulls()
        return result if not result.is_empty() else None
    except Exception as e:
        print(f"  Failed to download {ticker1}/{ticker2}: {e}")
        return None


def run_single_backtest(
    data: pl.DataFrame,
    params: dict,
) -> dict[str, float] | None:
    """
    Run a single backtest without saving to the database.

    Calls generate_signals and _calculate_returns directly to
    bypass Backtester.run() which triggers save_results().
    """
    strategy = PairsTradingStrategy(params)
    backtester = Backtester(data, strategy)
    signals = strategy.generate_signals(data)
    backtester.results = backtester._calculate_returns(signals)
    return backtester.get_performance_metrics()


def main():
    pairs = list(itertools.combinations(TICKERS, 2))
    param_combos = list(itertools.product(*PARAM_RANGES.values()))
    param_keys = list(PARAM_RANGES.keys())
    total_backtests = len(pairs) * len(param_combos)
    print(
        f"Pairs: {len(pairs)}, "
        f"params per pair: {len(param_combos)}, "
        f"total backtests per run: {total_backtests}"
    )

    # Pre-download data for all pairs (not timed).
    print("\nPre-downloading data...")
    pair_data: dict[tuple[str, str], pl.DataFrame] = {}
    for i, (t1, t2) in enumerate(pairs):
        print(f"  Downloading {i + 1}/{len(pairs)}: {t1} vs {t2}")
        data = download_pair_data(t1, t2)
        if data is not None:
            pair_data[(t1, t2)] = data
    print(f"\nLoaded data for {len(pair_data)}/{len(pairs)} pairs\n")

    # Benchmark runs.
    runtimes: list[float] = []
    for run_num in range(NUM_RUNS):
        start = time.perf_counter()
        best_sharpe = float("-inf")

        for data in pair_data.values():
            for combo in param_combos:
                params = dict(zip(param_keys, combo))
                metrics = run_single_backtest(data, params)
                if metrics and metrics["Sharpe Ratio"] > best_sharpe:
                    best_sharpe = metrics["Sharpe Ratio"]

        elapsed = time.perf_counter() - start
        runtimes.append(elapsed)
        print(f"Run {run_num + 1}/{NUM_RUNS}: {elapsed:.4f}s")

    # Write results.
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["runtime_seconds"])
        for t in runtimes:
            writer.writerow([f"{t:.4f}"])
    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
