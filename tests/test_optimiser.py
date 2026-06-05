"""
Contains tests for optimisation functions.
"""

import datetime
import math
from typing import Any

import polars as pl
import pytest
from quant_trading_strategy_backtester.app import (
    _get_final_backtest_data,
    _get_final_backtest_results,
    prepare_buy_and_hold_strategy_with_optimisation,
    prepare_pairs_trading_strategy_with_optimisation,
    prepare_single_ticker_strategy_with_optimisation,
)
from quant_trading_strategy_backtester.optimiser import (
    _split_data,
    get_validation_data,
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    optimise_single_ticker_strategy_ticker,
    optimise_strategy_params,
    run_backtest,
    run_optimisation,
    walk_forward_optimise,
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
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_two_tickers",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
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


def test_optimise_pairs_trading_tickers_ranks_fixed_pairs_on_train(monkeypatch):
    """Verify fixed-parameter pair selection ignores held-out test metrics."""
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    mock_polars_data = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": [100.0 + i for i in range(10)],
            "Close_2": [200.0 + i for i in range(10)],
        }
    )
    strategy_params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    train_scores = {
        ("AAPL", "GOOGL"): 1.0,
        ("AAPL", "MSFT"): 2.0,
        ("GOOGL", "MSFT"): 0.5,
    }
    test_scores = {
        ("AAPL", "GOOGL"): 99.0,
        ("AAPL", "MSFT"): -1.0,
        ("GOOGL", "MSFT"): 0.0,
    }

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_backtest(data, strategy_type, params, tickers):
        pair = tuple(tickers)
        scores = train_scores if len(data) == 7 else test_scores
        return None, {
            "Sharpe Ratio": scores[pair],
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_two_tickers",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    best_pair, best_params, metrics = optimise_pairs_trading_tickers(
        mock_top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 12, 31),
        strategy_params,
        optimise=False,
    )

    assert best_pair == ("AAPL", "MSFT")
    assert best_params == strategy_params
    assert metrics["Sharpe Ratio"] == -1.0


def test_optimise_pairs_trading_tickers_ranks_optimised_pairs_on_train(
    monkeypatch,
):
    """Verify optimised pair selection does not pass test data into grid search."""
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    mock_polars_data = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": [100.0 + i for i in range(10)],
            "Close_2": [200.0 + i for i in range(10)],
        }
    )
    strategy_params = {
        "window": [20, 30],
        "entry_z_score": [2.0],
        "exit_z_score": [0.5],
    }
    train_scores = {
        ("AAPL", "GOOGL"): 1.0,
        ("AAPL", "MSFT"): 2.0,
        ("GOOGL", "MSFT"): 0.5,
    }

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_optimise_strategy_params(
        data,
        strategy_type,
        parameter_ranges,
        tickers,
        test_data=None,
    ):
        assert test_data is None
        assert len(data) == 7
        pair = tuple(tickers)
        return {
            "window": 20,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
        }, {
            "Sharpe Ratio": train_scores[pair],
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    def mock_run_backtest(data, strategy_type, params, tickers):
        assert len(data) == 3
        return None, {
            "Sharpe Ratio": -1.0,
            "Total Return": 0.01,
            "Max Drawdown": -0.02,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_two_tickers",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.optimise_strategy_params",
        mock_optimise_strategy_params,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    best_pair, best_params, metrics = optimise_pairs_trading_tickers(
        mock_top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 12, 31),
        strategy_params,
        optimise=True,
    )

    assert best_pair == ("AAPL", "MSFT")
    assert best_params == {
        "window": 20,
        "entry_z_score": 2.0,
        "exit_z_score": 0.5,
    }
    assert metrics["Sharpe Ratio"] == -1.0


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
        raise AssertionError("Pair optimisation should not run a second grid search")

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


def test_handle_pairs_trading_walk_forward_uses_original_ranges(monkeypatch):
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
            {"Sharpe Ratio": 1.5, "Total Return": 0.2, "Max Drawdown": -0.1},
        )

    def mock_load_data(*args, **kwargs):
        return mock_polars_data

    def mock_run_optimisation(
        data,
        strategy_type,
        received_params,
        start_date,
        end_date,
        tickers,
        walk_forward=False,
    ):
        assert received_params == strategy_params
        assert walk_forward is True
        return (
            {"window": 25, "entry_z_score": 2.5, "exit_z_score": 0.7},
            {"Sharpe Ratio": 1.8, "Total Return": 0.3, "Max Drawdown": -0.1},
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

    _, ticker_display, optimised_params = (
        prepare_pairs_trading_strategy_with_optimisation(
            start_date, end_date, strategy_params, True, walk_forward=True
        )
    )

    assert ticker_display == "AAPL vs. GOOGL"
    assert optimised_params == {
        "window": 25,
        "entry_z_score": 2.5,
        "exit_z_score": 0.7,
    }


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

    def mock_run_optimisation(*args, **kwargs):
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
        "quant_trading_strategy_backtester.app.run_optimisation",
        mock_run_optimisation,
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


def test_walk_forward_optimise():
    """
    Verify that walk-forward optimisation returns per-fold results with
    out-of-sample metrics and stable structure.
    """
    # 60 rows gives 10 rows per segment with n_folds=5.
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(60)]
    closes = [100.0 + i * 0.5 for i in range(60)]
    data = pl.DataFrame({"Date": dates, "Close": closes})

    parameter_ranges: dict[str, range | list[int | float]] = {
        "short_window": [3, 5],
        "long_window": [10, 15],
    }

    best_params, agg_metrics, fold_results = walk_forward_optimise(
        data,
        "Moving Average Crossover",
        parameter_ranges,
        tickers="TEST",
        n_folds=3,
    )

    # Check structure.
    assert isinstance(best_params, dict)
    assert set(best_params.keys()) == {"short_window", "long_window"}
    assert isinstance(agg_metrics, dict)
    assert "Total Return" in agg_metrics
    assert "Sharpe Ratio" in agg_metrics
    assert "Max Drawdown" in agg_metrics
    assert len(fold_results) == 3

    # Each fold should have the expected keys.
    for fold in fold_results:
        assert "fold" in fold
        assert "train_rows" in fold
        assert "test_rows" in fold
        assert "params" in fold
        assert "in_sample_sharpe" in fold
        assert "oos_metrics" in fold

    # Training window should expand across folds.
    assert fold_results[0]["train_rows"] < fold_results[1]["train_rows"]
    assert fold_results[1]["train_rows"] < fold_results[2]["train_rows"]

    # Best params should come from the final fold.
    assert best_params == fold_results[-1]["params"]


def test_walk_forward_optimise_insufficient_data():
    """
    Verify that walk-forward raises ValueError when data is too small
    for the requested number of folds.
    """
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0 + i for i in range(5)]})

    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward_optimise(
            data,
            "Moving Average Crossover",
            {"short_window": [3], "long_window": [10]},
            tickers="TEST",
            n_folds=5,
        )


def test_split_data():
    """Verify _split_data splits at the correct ratio."""
    data = pl.DataFrame({"Close": list(range(100))})
    train, test = _split_data(data, train_ratio=0.7)
    assert len(train) == 70
    assert len(test) == 30
    assert train["Close"][0] == 0
    assert train["Close"][-1] == 69
    assert test["Close"][0] == 70
    assert test["Close"][-1] == 99


def test_split_data_small_dataset():
    """Verify _split_data handles small datasets."""
    data = pl.DataFrame({"Close": [1, 2, 3]})
    train, test = _split_data(data, train_ratio=0.7)
    assert len(train) == 2
    assert len(test) == 1


def test_get_validation_data_standard_split():
    """Verify standard optimisation validation uses the 30% test split."""
    data = pl.DataFrame({"Close": list(range(100))})
    validation_data = get_validation_data(data)
    assert len(validation_data) == 30
    assert validation_data["Close"][0] == 70
    assert validation_data["Close"][-1] == 99


def test_get_validation_data_walk_forward():
    """Verify walk-forward validation uses the final test fold."""
    data = pl.DataFrame({"Close": list(range(60))})
    validation_data = get_validation_data(data, walk_forward=True, n_folds=5)
    assert len(validation_data) == 10
    assert validation_data["Close"][0] == 50
    assert validation_data["Close"][-1] == 59


def test_get_final_backtest_data_uses_validation_split():
    """Verify optimised app results use validation-period data."""
    data = pl.DataFrame({"Close": list(range(100))})
    validation_data = _get_final_backtest_data(
        data, use_validation_data=True, walk_forward=False
    )
    full_data = _get_final_backtest_data(
        data, use_validation_data=False, walk_forward=False
    )

    assert len(validation_data) == 30
    assert validation_data["Close"][0] == 70
    assert len(full_data) == 100


def test_get_final_backtest_results_preserves_context_and_resets_returns():
    """Verify validation display uses contextual results but validation returns."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    context_results = pl.DataFrame(
        {
            "Date": dates,
            "signal": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            "position_change": [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0],
            "gross_strategy_returns": [0.1] * 10,
            "transaction_costs": [0.01] * 10,
            "strategy_returns": [0.09] * 10,
            "trade_turnover": [0.0] * 10,
            "cumulative_transaction_costs": [0.01 * (i + 1) for i in range(10)],
            "gross_cumulative_returns": [99.0] * 10,
            "cumulative_returns": [99.0] * 10,
            "equity_curve": [9_999_999.0] * 10,
        }
    )
    validation_data = pl.DataFrame({"Date": dates[7:]})

    final_results = _get_final_backtest_results(
        context_results,
        validation_data,
        use_validation_data=True,
        initial_capital=100_000.0,
    )

    assert final_results["Date"].to_list() == dates[7:]
    assert final_results["signal"].to_list() == [-1.0, 0.0, 0.0]
    assert final_results["cumulative_returns"].to_list() == pytest.approx(
        [1.09, 1.1881, 1.295029]
    )
    assert final_results["gross_cumulative_returns"].to_list() == pytest.approx(
        [1.1, 1.21, 1.331]
    )
    assert final_results["cumulative_transaction_costs"].to_list() == pytest.approx(
        [0.01, 0.02, 0.03]
    )
    assert final_results["equity_curve"].to_list() == pytest.approx(
        [109_000.0, 118_810.0, 129_502.9]
    )


def test_optimise_strategy_params_returns_test_metrics(monkeypatch):
    """Verify optimise_strategy_params evaluates best params on test data."""
    train_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100 + i for i in range(21)],
        }
    )
    test_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 2, i) for i in range(1, 11)],
            "Close": [120 + i for i in range(10)],
        }
    )

    call_log = []

    def mock_run_backtest(data, strategy_type, params, tickers):
        call_log.append(("backtest", len(data)))
        # Return different metrics for train vs test
        if len(data) == 21:
            return None, {
                "Sharpe Ratio": 1.0,
                "Total Return": 0.1,
                "Max Drawdown": -0.05,
            }
        else:
            return None, {
                "Sharpe Ratio": 0.5,
                "Total Return": 0.05,
                "Max Drawdown": -0.03,
            }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    params, metrics = optimise_strategy_params(
        train_data,
        "Moving Average Crossover",
        {"short_window": [5], "long_window": [20]},
        "AAPL",
        test_data=test_data,
    )

    # Should have run backtests for each param combo on train + once on test
    assert len(call_log) == 2  # 1 train combo + 1 test eval
    assert call_log[0] == ("backtest", 21)  # train
    assert call_log[1] == ("backtest", 10)  # test
    # Metrics should be from test data
    assert metrics["Sharpe Ratio"] == 0.5
    assert metrics["Total Return"] == 0.05


def test_optimise_strategy_params_skips_nan_sharpe(monkeypatch):
    """Verify optimisation ignores parameter sets with undefined Sharpe."""
    data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100 + i for i in range(21)],
        }
    )

    def mock_run_backtest(data, strategy_type, params, tickers):
        sharpe = math.nan if params["short_window"] == 5 else 1.0
        return None, {
            "Sharpe Ratio": sharpe,
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    params, metrics = optimise_strategy_params(
        data,
        "Moving Average Crossover",
        {"short_window": [5, 10], "long_window": [20]},
        "AAPL",
    )

    assert params == {"short_window": 10, "long_window": 20}
    assert metrics["Sharpe Ratio"] == 1.0


def test_optimise_strategy_params_raises_when_all_sharpes_are_nan(monkeypatch):
    """Verify optimisation fails clearly when every candidate is untradeable."""
    data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100 + i for i in range(21)],
        }
    )

    def mock_run_backtest(data, strategy_type, params, tickers):
        return None, {
            "Sharpe Ratio": math.nan,
            "Total Return": 0.0,
            "Max Drawdown": 0.0,
            "Trade Events": 0.0,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    with pytest.raises(ValueError, match="Parameter optimisation failed"):
        optimise_strategy_params(
            data,
            "Moving Average Crossover",
            {"short_window": [5, 10], "long_window": [20]},
            "AAPL",
        )


def test_optimise_strategy_params_skips_invalid_parameter_combinations(
    monkeypatch,
) -> None:
    """Verify optimisation does not run backtests for invalid grid values."""
    data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100 + i for i in range(21)],
        }
    )
    evaluated_params = []

    def mock_run_backtest(data, strategy_type, params, tickers):
        evaluated_params.append(params)
        return None, {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    params, _ = optimise_strategy_params(
        data,
        "Moving Average Crossover",
        {"short_window": [10, 30], "long_window": [20]},
        "AAPL",
    )

    assert evaluated_params == [{"short_window": 10, "long_window": 20}]
    assert params == {"short_window": 10, "long_window": 20}


def test_optimise_strategy_params_raises_when_all_combinations_are_invalid() -> None:
    """Verify optimisation reports invalid grids before running backtests."""
    data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100 + i for i in range(21)],
        }
    )

    with pytest.raises(ValueError, match="No valid parameter combinations"):
        optimise_strategy_params(
            data,
            "Pairs Trading",
            {"window": [20], "entry_z_score": [1.0], "exit_z_score": [1.0]},
            ["AAPL", "MSFT"],
        )


def test_optimise_buy_and_hold_ticker_uses_train_test_split(monkeypatch):
    """Verify buy-and-hold ticker selection uses train for ranking, test for eval."""
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0)]

    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    mock_data = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0 + i for i in range(100)],
        }
    )

    def mock_load_data(*args, **kwargs):
        return mock_data

    split_calls = []
    original_split = _split_data

    def tracking_split(data, train_ratio=0.7):
        train, test = original_split(data, train_ratio)
        split_calls.append((len(train), len(test)))
        return train, test

    backtest_calls = []

    class MockBacktester:
        def __init__(self, data, strategy, tickers):
            self.tickers = tickers
            backtest_calls.append(len(data))

        def run(self):
            return None

        def get_performance_metrics(self):
            returns = {"AAPL": 0.3, "GOOGL": 0.2}
            return {
                "Total Return": returns.get(self.tickers, 0.1),
                "Sharpe Ratio": 1.5,
                "Max Drawdown": -0.1,
            }

    def mock_backtest_init(data, strategy, tickers):
        returns = {"AAPL": 0.3, "GOOGL": 0.2}
        assert tickers in returns
        return MockBacktester(data, strategy, tickers)

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.data.load_yfinance_data_one_ticker",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        mock_load_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser._split_data", tracking_split
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.Backtester", mock_backtest_init
    )

    best_ticker, params, metrics = optimise_buy_and_hold_ticker(
        mock_top_companies, datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)
    )

    assert best_ticker == "AAPL"
    assert params == {}
    assert metrics["Total Return"] == 0.3
    assert backtest_calls == [70, 70, 30]
    # Verify _split_data was called (train/test split happened)
    assert len(split_calls) >= 1
    train_size, test_size = split_calls[0]
    assert train_size + test_size == 100
    assert train_size == 70  # 70% of 100
    assert test_size == 30  # 30% of 100
