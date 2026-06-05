"""
Tests for the Mean Reversion strategy class.
"""

from datetime import date, timedelta

import polars as pl
import pytest
from quant_trading_strategy_backtester.strategies.mean_reversion import (
    MeanReversionStrategy,
)


def test_mean_reversion_strategy_initialisation() -> None:
    params = {"window": 20, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    assert strategy.window == 20
    assert strategy.std_dev == 2.0


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"window": 0, "std_dev": 2.0}, "window must be positive"),
        ({"window": 20, "std_dev": 0.0}, "std_dev must be positive"),
    ],
)
def test_mean_reversion_rejects_invalid_parameters(
    params: dict[str, float], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        MeanReversionStrategy(params)


def test_mean_reversion_strategy_generate_signals(
    mock_polars_data: pl.DataFrame,
) -> None:
    params = {"window": 5, "std_dev": 2.0}
    strategy = MeanReversionStrategy(params)
    signals = strategy.generate_signals(mock_polars_data)
    assert isinstance(signals, pl.DataFrame)
    EXPECTED_COLS = {
        "signal",
        "mean",
        "std",
        "upper_band",
        "lower_band",
        "position_change",
    }
    for col in EXPECTED_COLS:
        assert col in signals.columns
    assert signals["signal"].is_in([0.0, 1.0, -1.0]).all()


def test_mean_reversion_stays_flat_when_volatility_is_zero() -> None:
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(10)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0] * len(dates)})

    strategy = MeanReversionStrategy({"window": 5, "std_dev": 2.0})
    signals = strategy.generate_signals(data)

    assert (signals["std"].drop_nulls() == 0.0).all()
    assert (signals["signal"] == 0.0).all()
    assert (signals["position_change"] == 0.0).all()
