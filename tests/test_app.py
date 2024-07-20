import datetime

import pandas as pd
import pytest
from quant_trading_strategy_backtester.app import (
    load_yfinance_data,
    load_yfinance_data_two_tickers,
    run_backtest,
)


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


def test_load_yfinance_data_two_tickers(monkeypatch, mock_data):
    def mock_download(*args, **kwargs):
        return mock_data

    monkeypatch.setattr("yfinance.download", mock_download)

    data = load_yfinance_data_two_tickers(
        "AAPL", "MSFT", datetime.date(2020, 1, 1), datetime.date(2020, 1, 31)
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "Close_1" in data.columns
    assert "Close_2" in data.columns


@pytest.mark.parametrize(
    "strategy_type,params",
    [
        ("Moving Average Crossover", {"short_window": 5, "long_window": 20}),
        ("Mean Reversion", {"window": 5, "std_dev": 2.0}),
        ("Pairs Trading", {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}),
    ],
)
def test_run_backtest(mock_data, strategy_type, params):
    if strategy_type == "Pairs Trading":
        # Create mock data for two assets
        mock_data = pd.DataFrame(
            {
                "Close_1": mock_data["Close"],
                "Close_2": mock_data["Close"] * 1.1,  # Slightly different prices
            }
        )

    results, metrics = run_backtest(mock_data, strategy_type, params)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric in metrics


def test_run_backtest_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy type"):
        run_backtest(pd.DataFrame(), "Invalid Strategy", {})
