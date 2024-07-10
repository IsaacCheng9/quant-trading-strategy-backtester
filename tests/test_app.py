import datetime
import pandas as pd
import pytest
from quant_trading_strategy_backtester.app import load_yfinance_data, run_backtest


@pytest.fixture
def mock_data():
    dates = pd.date_range(start="1/1/2020", end="1/31/2020")
    return pd.DataFrame(
        {
            "Open": [100] * len(dates),
            "High": [110] * len(dates),
            "Low": [90] * len(dates),
            "Close": [105] * len(dates),
            "Volume": [1000000] * len(dates),
        },
        index=dates,
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


def test_run_backtest(mock_data):
    results, metrics = run_backtest(mock_data, 5, 20)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(metrics, dict)
    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
