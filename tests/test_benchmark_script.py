"""Tests for scripts/benchmark_pairs_trading.py."""

import datetime
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pandas as pd
import polars as pl
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "benchmark_pairs_trading.py"
)


def _load_benchmark_module() -> ModuleType:
    """Load the pairs trading benchmark script as an importable module."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_pairs_trading",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_valid_param_combinations_filters_invalid_combos() -> None:
    """Verify valid parameter grid has 170 combos with exit < entry z-score."""
    benchmark = _load_benchmark_module()

    combinations = benchmark.build_valid_param_combinations(benchmark.PARAM_RANGES)

    assert len(combinations) == 170
    for params in combinations:
        assert set(params.keys()) == {"window", "entry_z_score", "exit_z_score"}
        assert params["exit_z_score"] < params["entry_z_score"]


def test_run_single_backtest_succeeds_for_all_valid_param_combinations() -> None:
    """Verify every valid parameter combination executes cleanly on pair data."""
    benchmark = _load_benchmark_module()
    combinations = benchmark.build_valid_param_combinations(benchmark.PARAM_RANGES)

    num_rows = 220
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(num_rows)
    ]
    close_2 = [100.0 + i * 0.25 + math.sin(i / 7.0) * 2.0 for i in range(num_rows)]
    close_1 = [
        1.5 * price + 10.0 + math.cos(i / 5.0) * 3.0 for i, price in enumerate(close_2)
    ]
    pair_data = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": close_1,
            "Close_2": close_2,
        }
    )

    for params in combinations:
        metrics = benchmark.run_single_backtest(pair_data, params)
        assert isinstance(metrics, dict)
        assert "Sharpe Ratio" in metrics
        assert "Total Return" in metrics


def test_download_pair_data_preserves_order_and_uses_auto_adjust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify download_pair_data uses auto_adjust=True and preserves order."""
    benchmark = _load_benchmark_module()
    dates = pd.date_range(start="2020-01-01", end="2020-01-03")
    captured_kwargs: dict[str, object] = {}

    def mock_download(*args, **kwargs):
        assert args[0] == ["TSLA", "MSFT"]
        captured_kwargs.update(kwargs)
        # Simulate yfinance sorting ticker columns alphabetically.
        columns = pd.MultiIndex.from_tuples([("Close", "MSFT"), ("Close", "TSLA")])
        return pd.DataFrame(
            [[200.0, 300.0], [201.0, 301.0], [202.0, 302.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr(benchmark.yf, "download", mock_download)

    result = benchmark.download_pair_data("TSLA", "MSFT")

    assert captured_kwargs["auto_adjust"] is True
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["Date", "Close_1", "Close_2"]
    assert result["Close_1"].to_list() == [300.0, 301.0, 302.0]
    assert result["Close_2"].to_list() == [200.0, 201.0, 202.0]


def test_download_pair_data_handles_ticker_first_multiindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify download_pair_data supports (ticker, 'Close') MultiIndex order."""
    benchmark = _load_benchmark_module()
    dates = pd.date_range(start="2020-01-01", end="2020-01-03")

    def mock_download(*args, **kwargs):
        assert kwargs["auto_adjust"] is True
        columns = pd.MultiIndex.from_tuples([("MSFT", "Close"), ("TSLA", "Close")])
        return pd.DataFrame(
            [[200.0, 300.0], [201.0, 301.0], [202.0, 302.0]],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr(benchmark.yf, "download", mock_download)

    result = benchmark.download_pair_data("TSLA", "MSFT")

    assert isinstance(result, pl.DataFrame)
    assert result["Close_1"].to_list() == [300.0, 301.0, 302.0]
    assert result["Close_2"].to_list() == [200.0, 201.0, 202.0]
