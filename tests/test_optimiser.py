"""
Contains tests for optimisation functions.
"""

import datetime
from typing import Any

import polars as pl
import pytest
from quant_trading_strategy_backtester.app import (
    prepare_buy_and_hold_strategy_with_optimisation,
    prepare_pairs_trading_strategy_with_optimisation,
    prepare_single_ticker_strategy_with_optimisation,
)
from quant_trading_strategy_backtester.optimiser import (
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    optimise_single_ticker_strategy_ticker,
    run_backtest,
    run_optimisation,
)


def test_optimise_buy_and_hold_ticker(monkeypatch):
    # Mock data and functions
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_backtest(*args, **kwargs):
        return None, {"Total Return": 0.3, "Sharpe Ratio": 1.5, "Max Drawdown": -0.1}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.data.load_yfinance_data_one_ticker",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)

    best_ticker, params, metrics = optimise_buy_and_hold_ticker(
        mock_top_companies, start_date, end_date
    )

    assert isinstance(best_ticker, str)
    assert best_ticker in [company[0] for company in mock_top_companies]
    assert isinstance(params, dict)
    assert len(params) == 0  # Buy and Hold has no parameters
    assert isinstance(metrics, dict)
    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics


def test_run_optimisation_buy_and_hold(monkeypatch):
    # Mock data and functions
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]

    def mock_get_top_companies(*args, **kwargs):
        return mock_top_companies

    def mock_optimise_buy_and_hold(*args, **kwargs):
        return (
            "AAPL",
            {},
            {"Total Return": 0.3, "Sharpe Ratio": 1.5, "Max Drawdown": -0.1},
        )

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.get_top_sp500_companies",
        mock_get_top_companies,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.optimise_buy_and_hold_ticker",
        mock_optimise_buy_and_hold,
    )

    strategy_type = "Buy and Hold"
    initial_params = {}  # Buy and Hold has no parameters
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)

    optimised_params, metrics = run_optimisation(
        mock_polars_data, strategy_type, initial_params, start_date, end_date, "AAPL"
    )

    assert isinstance(optimised_params, dict)
    assert len(optimised_params) == 0  # Buy and Hold has no parameters
    assert isinstance(metrics, dict)
    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics


def test_prepare_buy_and_hold_strategy_with_optimisation(monkeypatch):
    # Mock data and functions
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]

    def mock_get_top_companies(*args, **kwargs):
        return mock_top_companies

    def mock_optimise_buy_and_hold(*args, **kwargs):
        return (
            "AAPL",
            {},
            {"Total Return": 0.3, "Sharpe Ratio": 1.5, "Max Drawdown": -0.1},
        )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_top_sp500_companies",
        mock_get_top_companies,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.optimise_buy_and_hold_ticker",
        mock_optimise_buy_and_hold,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        mock_load_data,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    data, ticker_display, strategy_params = (
        prepare_buy_and_hold_strategy_with_optimisation(start_date, end_date)
    )

    assert isinstance(data, pl.DataFrame)
    assert isinstance(ticker_display, str)
    assert ticker_display == "AAPL"
    assert isinstance(strategy_params, dict)
    assert len(strategy_params) == 0  # Buy and Hold has no parameters


def test_optimise_pairs_trading_tickers(monkeypatch):
    # Mock data and functions
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    mock_polars_data = pl.DataFrame(
        {"Close_1": [100, 101, 102], "Close_2": [200, 202, 204]}
    )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_backtest(*args, **kwargs):
        return None, {"Sharpe Ratio": 1.5}

    def mock_optimise_strategy_params(*args, **kwargs):
        return {"window": 25, "entry_z_score": 2.5, "exit_z_score": 0.6}, {
            "Sharpe Ratio": 1.8
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.data.load_yfinance_data_two_tickers",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.optimise_strategy_params",
        mock_optimise_strategy_params,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    strategy_params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    # Test with optimisation
    best_pair, best_params, _ = optimise_pairs_trading_tickers(
        mock_top_companies, start_date, end_date, strategy_params, True
    )

    assert isinstance(best_pair, tuple)
    assert len(best_pair) == 2
    assert all(
        ticker in [company[0] for company in mock_top_companies] for ticker in best_pair
    )
    assert isinstance(best_params, dict)
    assert set(best_params.keys()) == set(strategy_params.keys())
    assert best_params["window"] == 25  # Optimised value

    # Test without optimisation
    best_pair, best_params, _ = optimise_pairs_trading_tickers(
        mock_top_companies, start_date, end_date, strategy_params, False
    )

    assert isinstance(best_pair, tuple)
    assert len(best_pair) == 2
    assert all(
        ticker in [company[0] for company in mock_top_companies] for ticker in best_pair
    )
    assert isinstance(best_params, dict)
    # Should be the same as input when not optimising
    assert best_params == strategy_params


def test_handle_pairs_trading_optimisation(monkeypatch):
    # Mock data and functions
    mock_polars_data = pl.DataFrame(
        {"Close_1": [100, 101, 102], "Close_2": [200, 202, 204]}
    )
    mock_top_companies = [("AAPL", 1000000), ("GOOGL", 900000), ("MSFT", 800000)]

    def mock_get_top_companies(*args, **kwargs):
        return mock_top_companies

    def mock_optimise_pairs(*args, **kwargs):
        return (
            ("AAPL", "GOOGL"),
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            None,
        )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_optimisation(*args, **kwargs):
        return (
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            {"Sharpe Ratio": 1.5, "Total Return": 0.2, "Max Drawdown": -0.1},
        )

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_top_sp500_companies",
        mock_get_top_companies,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.optimise_pairs_trading_tickers",
        mock_optimise_pairs,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_two_tickers",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.run_optimisation",
        mock_run_optimisation,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    strategy_params = {
        "window": range(10, 31),
        "entry_z_score": [1.5, 2.0, 2.5],
        "exit_z_score": [0.3, 0.5, 0.7],
    }

    data, ticker_display, optimised_params = (
        prepare_pairs_trading_strategy_with_optimisation(
            start_date, end_date, strategy_params, True
        )
    )

    assert isinstance(data, pl.DataFrame)
    assert ticker_display == "AAPL vs. GOOGL"
    assert isinstance(optimised_params, dict)
    assert set(optimised_params.keys()) == set(strategy_params.keys())
    assert optimised_params["window"] == 20
    assert optimised_params["entry_z_score"] == 2.0
    assert optimised_params["exit_z_score"] == 0.5


def test_run_optimisation(monkeypatch):
    # Mock data and functions
    mock_polars_data = pl.DataFrame({"Close": [100, 101, 102]})

    def mock_optimise_strategy_params(*args, **kwargs):
        return {"window": 25, "std_dev": 2.5}, {"Sharpe Ratio": 1.8}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.optimise_strategy_params",
        mock_optimise_strategy_params,
    )

    strategy_type = "Mean Reversion"
    initial_params = {"window": 20, "std_dev": 2.0}
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    ticker = "AAPL"

    optimised_params, metrics = run_optimisation(
        mock_polars_data,
        strategy_type,
        initial_params,
        start_date,
        end_date,
        ticker,
    )

    assert isinstance(optimised_params, dict)
    assert set(optimised_params.keys()) == set(initial_params.keys())
    assert isinstance(metrics, dict)
    assert "Sharpe Ratio" in metrics


@pytest.mark.parametrize(
    "strategy_type,params,tickers",
    [
        ("Moving Average Crossover", {"short_window": 5, "long_window": 20}, "AAPL"),
        ("Mean Reversion", {"window": 5, "std_dev": 2.0}, "AAPL"),
        (
            "Pairs Trading",
            {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5},
            ["AAPL", "GOOGL"],
        ),
    ],
)
def test_run_backtest(
    mock_polars_data: pl.DataFrame,
    strategy_type: str,
    params: dict[str, Any],
    tickers: str | list[str],
) -> None:
    # Ensure mock_polars_data has a Date column
    if "Date" not in mock_polars_data.columns:
        mock_polars_data = mock_polars_data.with_columns(
            pl.date_range(
                start=datetime.date(2020, 1, 1),
                end=datetime.date(2020, 1, 31),
                interval="1d",
            ).alias("Date")
        )

    if strategy_type == "Pairs Trading":
        # Create mock data for two assets
        mock_polars_data = pl.DataFrame(
            {
                "Date": mock_polars_data["Date"],
                "Close_1": mock_polars_data["Close"],
                "Close_2": mock_polars_data["Close"] * 1.1,  # Slightly different prices
            }
        )
    elif "Close" not in mock_polars_data.columns:
        mock_polars_data = mock_polars_data.with_columns(pl.col("Open").alias("Close"))

    results, metrics = run_backtest(mock_polars_data, strategy_type, params, tickers)
    assert isinstance(results, pl.DataFrame)
    assert isinstance(metrics, dict)
    EXPECTED_METRICS = {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for metric in EXPECTED_METRICS:
        assert metric in metrics


def test_run_backtest_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="Invalid strategy type"):
        run_backtest(pl.DataFrame(), "Invalid Strategy", {}, "AAPL")


def test_optimise_single_ticker_strategy_ticker(monkeypatch):
    # Mock data and functions
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_backtest(*args, **kwargs):
        strategy_type = args[1]
        if strategy_type == "Moving Average Crossover":
            return None, {"Sharpe Ratio": 1.5}
        elif strategy_type == "Mean Reversion":
            return None, {"Sharpe Ratio": 1.2}
        else:
            return None, {"Sharpe Ratio": 1.0}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        mock_run_backtest,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    strategy_type = "Moving Average Crossover"
    strategy_params = {"short_window": 10, "long_window": 50}

    best_ticker = optimise_single_ticker_strategy_ticker(
        mock_top_companies, start_date, end_date, strategy_type, strategy_params
    )

    assert isinstance(best_ticker, str)
    assert best_ticker in [company[0] for company in mock_top_companies]


def test_prepare_single_ticker_strategy_with_optimisation(monkeypatch):
    # Mock data and functions
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]

    def mock_get_top_companies(*args, **kwargs):
        return mock_top_companies

    def mock_optimise_single_ticker(*args, **kwargs):
        return "AAPL"

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_optimise_strategy_params(*args, **kwargs):
        return {"short_window": 15, "long_window": 60}, {"Sharpe Ratio": 1.8}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_top_sp500_companies",
        mock_get_top_companies,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.optimise_single_ticker_strategy_ticker",
        mock_optimise_single_ticker,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.optimise_strategy_params",
        mock_optimise_strategy_params,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    strategy_type = "Moving Average Crossover"
    strategy_params = {
        "short_window": range(5, 30, 5),
        "long_window": range(20, 100, 10),
    }
    optimise = True

    data, ticker_display, optimised_params = (
        prepare_single_ticker_strategy_with_optimisation(
            start_date, end_date, strategy_type, strategy_params, optimise
        )
    )

    assert isinstance(data, pl.DataFrame)
    assert isinstance(ticker_display, str)
    assert ticker_display == "AAPL"
    assert isinstance(optimised_params, dict)
    assert set(optimised_params.keys()) == {"short_window", "long_window"}
    assert optimised_params["short_window"] == 15
    assert optimised_params["long_window"] == 60


def test_prepare_single_ticker_strategy_with_optimisation_no_param_optimisation(
    monkeypatch,
):
    # Mock data and functions
    mock_polars_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100 + i for i in range(31)],
        }
    )
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]

    def mock_get_top_companies(*args, **kwargs):
        return mock_top_companies

    def mock_optimise_single_ticker(*args, **kwargs):
        return "AAPL"

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_optimise_strategy_params(*args, **kwargs):
        return {"short_window": 15, "long_window": 60}, {"Sharpe Ratio": 1.8}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.get_top_sp500_companies",
        mock_get_top_companies,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.optimise_single_ticker_strategy_ticker",
        mock_optimise_single_ticker,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.app.load_yfinance_data_one_ticker",
        mock_load_data,
    )

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    strategy_type = "Moving Average Crossover"
    strategy_params = {"short_window": 10, "long_window": 50}
    optimise = False

    data, ticker_display, final_params = (
        prepare_single_ticker_strategy_with_optimisation(
            start_date, end_date, strategy_type, strategy_params, optimise
        )
    )

    assert isinstance(data, pl.DataFrame)
    assert isinstance(ticker_display, str)
    assert ticker_display == "AAPL"
    assert isinstance(final_params, dict)
    assert final_params == strategy_params  # Parameters should remain unchanged
