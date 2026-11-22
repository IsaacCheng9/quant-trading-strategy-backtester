"""Benchmark pairs trading with parameter optimisation.

Pre-downloads data for all ticker pairs, then times only the computation
(signal generation + return calculation + metrics) over all pairs and valid
parameter combinations. This isolates the Polars computation from network I/O.
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
from quant_trading_strategy_backtester.strategy_params import (
    is_valid_strategy_params,
)

# Pinned top 20 S&P 500 companies by market cap for consistent benchmarks
# across runs.
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
PARAM_RANGES: dict[str, list[int | float]] = {
    "window": list(range(10, 101, 10)),
    "entry_z_score": [1.0, 1.5, 2.0, 2.5, 3.0],
    "exit_z_score": [0.1, 0.5, 1.0, 1.5],
}
NUM_RUNS = 6
OUTPUT_CSV = "resources/polars_pairs_trading_benchmark.csv"


def build_valid_param_combinations(
    param_ranges: dict[str, list[int | float]],
) -> list[dict[str, int | float]]:
    """Build valid pairs trading parameter combinations from range definitions.

    Args:
        param_ranges: Mapping of parameter names to candidate values.

    Returns:
        A list of parameter dictionaries accepted by PairsTradingStrategy.
    """
    param_keys = list(param_ranges.keys())
    valid_combinations: list[dict[str, int | float]] = []
    for values in itertools.product(*param_ranges.values()):
        params = dict(zip(param_keys, values, strict=True))
        if is_valid_strategy_params("Pairs Trading", params):
            valid_combinations.append(params)
    return valid_combinations


def _select_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Select a ticker's close series from either yfinance MultiIndex layout.

    Args:
        data: The yfinance response for multiple tickers.
        ticker: The ticker whose close prices should be selected.

    Returns:
        The ticker's close-price series.

    Raises:
        ValueError: If the response does not contain exactly one close column.
    """
    matches = [
        column
        for column in data.columns
        if isinstance(column, tuple) and "Close" in column and ticker in column
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Close column for {ticker}, found {len(matches)}"
        )

    return data[matches[0]]


def download_pair_data(ticker1: str, ticker2: str) -> pl.DataFrame | None:
    """Download and format adjusted price data for a ticker pair.

    Args:
        ticker1: The first ticker symbol (mapped to Close_1).
        ticker2: The second ticker symbol (mapped to Close_2).

    Returns:
        A Polars DataFrame with Date, Close_1, and Close_2 columns, or None if
        the download fails or returns empty data.
    """
    try:
        raw = yf.download(
            [ticker1, ticker2],
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
        )
        raw = cast(pd.DataFrame, raw)
        if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            return None

        close_1 = _select_close_series(raw, ticker1)
        close_2 = _select_close_series(raw, ticker2)
        close_data = pd.DataFrame(
            {
                "Date": raw.index,
                "Close_1": close_1.to_numpy(),
                "Close_2": close_2.to_numpy(),
            }
        )
        result = pl.from_pandas(close_data).drop_nulls()
        return result if not result.is_empty() else None
    except Exception as exc:  # noqa: BLE001
        print(f"  Failed to download {ticker1}/{ticker2}: {exc}")
        return None


def run_single_backtest(
    data: pl.DataFrame,
    params: dict[str, int | float],
) -> dict[str, float] | None:
    """Run a single pairs trading backtest without persisting results.

    Args:
        data: Polars DataFrame with Date, Close_1, and Close_2 columns.
        params: Validated parameter dictionary for PairsTradingStrategy.

    Returns:
        Dictionary of performance metrics, or None if results are unavailable.
    """
    strategy = PairsTradingStrategy(params)
    backtester = Backtester(data, strategy)
    backtester.run()
    return backtester.get_performance_metrics()


def main() -> None:
    """Benchmark pairs trading across all ticker pairs and valid parameters."""
    pairs = list(itertools.combinations(TICKERS, 2))
    param_combos = build_valid_param_combinations(PARAM_RANGES)
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
            for params in param_combos:
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
