"""
Tests for pair cointegration diagnostics.
"""

import numpy as np
import polars as pl
import pytest

from quant_trading_strategy_backtester.cointegration import evaluate_cointegration


def test_cointegration_accepts_stationary_linear_relationship() -> None:
    """Verify Engle-Granger accepts a stationary pair relationship."""
    rng = np.random.default_rng(42)
    close_2 = np.cumsum(rng.normal(0, 1, 300)) + 100
    close_1 = 5 + 2 * close_2 + rng.normal(0, 0.5, 300)
    data = pl.DataFrame({"Close_1": close_1, "Close_2": close_2})

    result = evaluate_cointegration(data)

    assert result.is_cointegrated
    assert result.p_value < 0.05
    assert result.reason is None


def test_cointegration_rejects_independent_random_walks() -> None:
    """Verify Engle-Granger rejects unrelated random walks."""
    rng = np.random.default_rng(12)
    close_1 = np.cumsum(rng.normal(0, 1, 300)) + 100
    close_2 = np.cumsum(rng.normal(0, 1, 300)) + 50
    data = pl.DataFrame({"Close_1": close_1, "Close_2": close_2})

    result = evaluate_cointegration(data)

    assert not result.is_cointegrated
    assert result.p_value > 0.05


def test_cointegration_rejects_short_series() -> None:
    """Verify short samples are rejected rather than force-ranked."""
    data = pl.DataFrame({"Close_1": [1.0, 2.0], "Close_2": [2.0, 3.0]})

    result = evaluate_cointegration(data)

    assert not result.is_cointegrated
    assert result.reason == "Not enough observations for cointegration test"


def test_cointegration_requires_pair_close_columns() -> None:
    """Verify missing pair columns fail clearly."""
    data = pl.DataFrame({"Close": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Close_1 and Close_2"):
        evaluate_cointegration(data)
