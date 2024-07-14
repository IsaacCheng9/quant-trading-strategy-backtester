import datetime

import pandas as pd
import pytest
from quant_trading_strategy_backtester.app import load_yfinance_data, run_backtest


def test_load_yfinance_data(monkeypatch, mock_data):
    def mock_download(*args, **kwargs):
        return mock_data

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data(
        "AAPL", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "Close" in data.columns


@pytest.mark.parametrize(
    "strategy_type", ["Moving Average Crossover", "Mean Reversion"]
)
def test_run_backtest(mock_data, strategy_type):
    params = (
        {"short_window": 5, "long_window": 20}
        if strategy_type == "Moving Average Crossover"
        else {"window": 5, "std_dev": 2.0}
    )
    results, metrics = run_backtest(mock_data, strategy_type, **params)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric in metrics
