"""
Contains tests for optimisation functions.
"""

import datetime
import math
import subprocess
import sys
from typing import Any

import polars as pl
import pytest

from quant_trading_strategy_backtester.app import (
    _get_final_backtest_data,
    prepare_buy_and_hold_strategy_with_optimisation,
    prepare_pairs_trading_strategy_with_optimisation,
    prepare_single_ticker_strategy_with_optimisation,
)
from quant_trading_strategy_backtester.backtester import (
    TRADING_DAYS_PER_YEAR,
    build_trade_ledger,
)
from quant_trading_strategy_backtester.cointegration import CointegrationResult
from quant_trading_strategy_backtester.optimiser import (
    PortfolioPosition,
    _extract_closing_position,
    _split_data,
    count_candidate_pairs,
    count_valid_parameter_combinations,
    evaluate_validation_period,
    get_training_data,
    get_validation_data,
    get_walk_forward_oos_data,
    optimise_buy_and_hold_ticker,
    optimise_pairs_trading_tickers,
    optimise_single_ticker_strategy_ticker,
    optimise_strategy_params,
    run_backtest,
    run_optimisation,
    stitch_walk_forward_oos_results,
    walk_forward_optimise,
)


def _cointegration_result(is_cointegrated: bool) -> CointegrationResult:
    """Return a deterministic cointegration result for optimiser tests."""
    return CointegrationResult(
        is_cointegrated=is_cointegrated,
        p_value=0.01 if is_cointegrated else 0.5,
        test_statistic=-4.0 if is_cointegrated else -1.0,
        critical_value_1pct=-3.9,
        critical_value_5pct=-3.3,
        critical_value_10pct=-3.0,
        reason=None if is_cointegrated else "Pair is not cointegrated",
    )


def test_count_valid_parameter_combinations_filters_invalid_combinations() -> None:
    """Verify optimisation reporting counts valid grid candidates only."""
    assert (
        count_valid_parameter_combinations(
            "Moving Average Crossover",
            {"short_window": [10, 30], "long_window": [20]},
        )
        == 1
    )
    assert (
        count_valid_parameter_combinations(
            "Moving Average Crossover",
            {"short_window": 10, "long_window": 20},
        )
        == 1
    )


def test_count_candidate_pairs_excludes_same_company_pairs(monkeypatch) -> None:
    """Verify pair reporting counts candidates after same-company filtering."""
    top_companies = [("GOOG", 1.0), ("GOOGL", 0.9), ("MSFT", 0.8)]

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda ticker1, ticker2: {ticker1, ticker2} == {"GOOG", "GOOGL"},
    )

    assert count_candidate_pairs(top_companies) == 2


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
    # Mock data and functions.
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    mock_batch_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, day) for day in range(1, 4)],
            "AAPL": [100.0, 101.0, 102.0],
            "GOOGL": [200.0, 202.0, 204.0],
            "MSFT": [300.0, 303.0, 306.0],
        }
    )

    def mock_load_many_data(*args, **kwargs):
        return mock_batch_data

    def mock_run_backtest(*args, **kwargs):
        return None, {"Sharpe Ratio": 1.5}

    def mock_optimise_strategy_params(*args, **kwargs):
        return {"window": 25, "entry_z_score": 2.5, "exit_z_score": 0.6}, {
            "Sharpe Ratio": 1.8
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        mock_load_many_data,
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
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        lambda *_args, **_kwargs: (pl.DataFrame(), {"Sharpe Ratio": 1.5}),
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
    mock_batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i for i in range(10)],
            "GOOGL": [200.0 + i for i in range(10)],
            "MSFT": [300.0 + i for i in range(10)],
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

    def mock_load_many_data(*args, **kwargs):
        return mock_batch_data

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
        pair = tuple(tickers)
        return None, {
            "Sharpe Ratio": train_scores[pair],
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    def mock_evaluate_validation_period(
        context_data,
        validation_data,
        strategy_type,
        params,
        tickers,
        **_kwargs,
    ):
        assert len(context_data) == 10
        assert len(validation_data) == 3
        return pl.DataFrame(), {
            "Sharpe Ratio": test_scores[tuple(tickers)],
            "Total Return": 0.01,
            "Max Drawdown": -0.02,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        mock_load_many_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        mock_evaluate_validation_period,
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
    mock_batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i for i in range(10)],
            "GOOGL": [200.0 + i for i in range(10)],
            "MSFT": [300.0 + i for i in range(10)],
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

    def mock_load_many_data(*args, **kwargs):
        return mock_batch_data

    def mock_optimise_strategy_params(
        data,
        strategy_type,
        parameter_ranges,
        tickers,
        test_data=None,
        **_kwargs,
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

    def mock_evaluate_validation_period(
        context_data,
        validation_data,
        strategy_type,
        params,
        tickers,
        **_kwargs,
    ):
        assert len(context_data) == 10
        assert len(validation_data) == 3
        return pl.DataFrame(), {
            "Sharpe Ratio": -1.0,
            "Total Return": 0.01,
            "Max Drawdown": -0.02,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        mock_load_many_data,
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
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        mock_evaluate_validation_period,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
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


def test_optimise_pairs_trading_tickers_filters_non_cointegrated_pairs(
    monkeypatch,
) -> None:
    """Verify automatic pair selection rejects non-cointegrated candidates."""
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0), ("MSFT", 800000.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    mock_batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i * 0.1 for i in range(100)],
            "GOOGL": [200.0 + i * 0.2 for i in range(100)],
            "MSFT": [300.0 + i * 0.3 for i in range(100)],
        }
    )
    strategy_params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}
    train_scores = {
        ("AAPL", "GOOGL"): 99.0,
        ("AAPL", "MSFT"): 2.0,
        ("GOOGL", "MSFT"): 1.0,
    }

    def mock_evaluate_cointegration(data, **_kwargs):
        is_aapl_googl = data["Close_1"][0] == 100.0 and data["Close_2"][0] == 200.0
        return _cointegration_result(not is_aapl_googl)

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
        pair = tuple(tickers)
        return None, {
            "Sharpe Ratio": train_scores[pair],
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    def mock_evaluate_validation_period(
        context_data,
        validation_data,
        strategy_type,
        params,
        tickers,
        **_kwargs,
    ):
        pair = tuple(tickers)
        return pl.DataFrame(), {
            "Sharpe Ratio": train_scores[pair],
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        lambda *_args, **_kwargs: mock_batch_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        mock_evaluate_cointegration,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        mock_evaluate_validation_period,
    )

    best_pair, _, metrics = optimise_pairs_trading_tickers(
        mock_top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 12, 31),
        strategy_params,
        optimise=False,
    )

    assert best_pair == ("AAPL", "MSFT")
    assert metrics["Sharpe Ratio"] == 2.0


def test_optimise_pairs_trading_tickers_raises_when_all_pairs_fail_filter(
    monkeypatch,
) -> None:
    """Verify pair selection fails clearly when no candidate is cointegrated."""
    mock_top_companies = [("AAPL", 1000000.0), ("GOOGL", 900000.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    mock_batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i * 0.1 for i in range(100)],
            "GOOGL": [200.0 + i * 0.2 for i in range(100)],
        }
    )
    strategy_params = {"window": 20, "entry_z_score": 2.0, "exit_z_score": 0.5}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        lambda *_args, **_kwargs: mock_batch_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(False),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        lambda *_args, **_kwargs: pytest.fail("Backtest should not run"),
    )

    with pytest.raises(ValueError, match="no cointegrated pair"):
        optimise_pairs_trading_tickers(
            mock_top_companies,
            datetime.date(2020, 1, 1),
            datetime.date(2020, 12, 31),
            strategy_params,
            optimise=False,
        )


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


def test_automatic_ticker_selection_rejects_walk_forward():
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    single_ticker_params = {
        "window": range(5, 21, 5),
        "std_dev": [1.0, 2.0],
    }
    pairs_params = {
        "window": range(10, 31),
        "entry_z_score": [1.5, 2.0, 2.5],
        "exit_z_score": [0.3, 0.5, 0.7],
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_single_ticker_strategy_with_optimisation(
            start_date,
            end_date,
            "Mean Reversion",
            single_ticker_params,
            True,
            walk_forward=True,
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_pairs_trading_strategy_with_optimisation(
            start_date,
            end_date,
            pairs_params,
            True,
            walk_forward=True,
        )


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
        assert "oos_results" in fold
        assert isinstance(fold["oos_results"], pl.DataFrame)
        assert len(fold["oos_results"]) == fold["test_rows"]

    # Training window should expand across folds.
    assert fold_results[0]["train_rows"] < fold_results[1]["train_rows"]
    assert fold_results[1]["train_rows"] < fold_results[2]["train_rows"]

    # Best params should come from the final fold.
    assert best_params == fold_results[-1]["params"]

    # Returned metrics should match stitching the per-fold OOS results.
    stitched_results, stitched_metrics = stitch_walk_forward_oos_results(fold_results)
    assert len(stitched_results) == sum(fold["test_rows"] for fold in fold_results)
    assert agg_metrics == pytest.approx(stitched_metrics, nan_ok=True)


def _make_fold_oos_results(
    start_date: datetime.date,
    strategy_returns: list[float],
    trade_turnovers: list[float],
    cost_rate: float = 0.0008,
    initial_capital: float = 100000.0,
) -> pl.DataFrame:
    """Build synthetic per-fold OOS results for stitching tests."""
    dates = [
        start_date + datetime.timedelta(days=i) for i in range(len(strategy_returns))
    ]
    transaction_costs = [turnover * cost_rate for turnover in trade_turnovers]
    gross_returns = [
        net_return + cost
        for net_return, cost in zip(strategy_returns, transaction_costs, strict=True)
    ]
    frame = pl.DataFrame(
        {
            "Date": dates,
            "gross_strategy_returns": gross_returns,
            "strategy_returns": strategy_returns,
            "transaction_costs": transaction_costs,
            "trade_turnover": trade_turnovers,
        }
    )
    return frame.with_columns(
        [
            (1 + pl.col("gross_strategy_returns"))
            .cum_prod()
            .alias("gross_cumulative_returns"),
            (1 + pl.col("strategy_returns")).cum_prod().alias("cumulative_returns"),
            (initial_capital * (1 + pl.col("strategy_returns")).cum_prod()).alias(
                "equity_curve"
            ),
            pl.col("transaction_costs").cum_sum().alias("cumulative_transaction_costs"),
        ]
    )


def test_stitch_walk_forward_oos_results_compounds_total_return() -> None:
    """Verify stitched total return compounds returns instead of averaging."""
    fold1_returns = [0.10, 0.10]
    fold2_returns = [-0.10, 0.05]
    fold1 = _make_fold_oos_results(datetime.date(2020, 1, 1), fold1_returns, [1.0, 0.0])
    fold2 = _make_fold_oos_results(datetime.date(2020, 1, 3), fold2_returns, [1.0, 0.0])

    fold_results = [
        {"fold": 1, "oos_results": fold1},
        {"fold": 2, "oos_results": fold2},
    ]

    stitched_results, stitched_metrics = stitch_walk_forward_oos_results(fold_results)

    expected_total_return = math.prod(1 + r for r in fold1_returns + fold2_returns) - 1
    fold1_return = math.prod(1 + r for r in fold1_returns) - 1
    fold2_return = math.prod(1 + r for r in fold2_returns) - 1
    mean_fold_return = (fold1_return + fold2_return) / 2

    assert len(stitched_results) == 4
    assert stitched_metrics["Total Return"] == pytest.approx(expected_total_return)
    assert stitched_metrics["Total Return"] != pytest.approx(mean_fold_return)


def test_stitch_walk_forward_oos_results_captures_cross_fold_drawdown() -> None:
    """Verify stitched max drawdown captures cross-fold losses."""
    fold1 = _make_fold_oos_results(
        datetime.date(2020, 1, 1), [0.20, -0.05], [1.0, 0.0], cost_rate=0.0
    )
    fold2 = _make_fold_oos_results(
        datetime.date(2020, 1, 3), [-0.10, -0.10], [0.0, 0.0], cost_rate=0.0
    )

    _, fold1_metrics = stitch_walk_forward_oos_results(
        [{"fold": 1, "oos_results": fold1}]
    )
    _, fold2_metrics = stitch_walk_forward_oos_results(
        [{"fold": 2, "oos_results": fold2}]
    )
    _, stitched_metrics = stitch_walk_forward_oos_results(
        [{"fold": 1, "oos_results": fold1}, {"fold": 2, "oos_results": fold2}]
    )

    # Peak on day 1 is 1.20; trough on day 4 is 1.20 * 0.95 * 0.90 * 0.90.
    expected_cross_fold_dd = (0.95 * 0.90 * 0.90) - 1.0
    assert stitched_metrics["Max Drawdown"] == pytest.approx(expected_cross_fold_dd)
    assert abs(stitched_metrics["Max Drawdown"]) > abs(fold1_metrics["Max Drawdown"])
    assert abs(stitched_metrics["Max Drawdown"]) > abs(fold2_metrics["Max Drawdown"])


def test_stitch_walk_forward_oos_results_sums_trade_activity_and_costs() -> None:
    """Verify trade events, turnover, and costs sum across folds."""
    fold1 = _make_fold_oos_results(
        datetime.date(2020, 1, 1), [0.01, -0.005, 0.02], [1.0, 0.0, 2.0]
    )
    fold2 = _make_fold_oos_results(
        datetime.date(2020, 1, 4), [0.005, 0.01, -0.01], [1.5, 0.5, 1.0]
    )

    _, fold1_metrics = stitch_walk_forward_oos_results(
        [{"fold": 1, "oos_results": fold1}]
    )
    _, fold2_metrics = stitch_walk_forward_oos_results(
        [{"fold": 2, "oos_results": fold2}]
    )
    stitched_results, stitched_metrics = stitch_walk_forward_oos_results(
        [{"fold": 1, "oos_results": fold1}, {"fold": 2, "oos_results": fold2}]
    )

    assert stitched_metrics["Trade Events"] == pytest.approx(
        fold1_metrics["Trade Events"] + fold2_metrics["Trade Events"]
    )
    assert stitched_metrics["Trade Events"] == 5.0
    assert stitched_metrics["Total Turnover"] == pytest.approx(
        fold1_metrics["Total Turnover"] + fold2_metrics["Total Turnover"]
    )
    assert stitched_metrics["Total Costs"] == pytest.approx(
        fold1_metrics["Total Costs"] + fold2_metrics["Total Costs"]
    )
    assert stitched_results["cumulative_transaction_costs"][-1] == pytest.approx(
        stitched_metrics["Total Costs"]
    )


def test_stitch_walk_forward_oos_results_ignores_zero_trade_fold_nan_sharpe() -> None:
    """Verify a zero-trade fold NaN Sharpe does not poison stitched Sharpe."""
    active_fold = _make_fold_oos_results(
        datetime.date(2020, 1, 1), [0.01, 0.02, -0.005, 0.015], [1.0, 0.0, 0.0, 1.0]
    )
    flat_fold = _make_fold_oos_results(
        datetime.date(2020, 1, 5), [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    )

    _, flat_metrics = stitch_walk_forward_oos_results(
        [{"fold": 2, "oos_results": flat_fold}]
    )
    assert math.isnan(flat_metrics["Sharpe Ratio"])

    _, stitched_metrics = stitch_walk_forward_oos_results(
        [
            {"fold": 1, "oos_results": active_fold},
            {"fold": 2, "oos_results": flat_fold},
        ]
    )
    assert math.isfinite(stitched_metrics["Sharpe Ratio"])
    assert stitched_metrics["Sharpe Ratio"] > 0.0


def test_walk_forward_optimise_includes_trailing_remainder_rows() -> None:
    """Verify trailing remainder rows are assigned to the final fold."""
    n_rows = 63
    n_folds = 3
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    closes = [100.0 + i * 0.5 for i in range(n_rows)]
    data = pl.DataFrame({"Date": dates, "Close": closes})

    _, _, fold_results = walk_forward_optimise(
        data,
        "Moving Average Crossover",
        {"short_window": [3, 5], "long_window": [10, 15]},
        tickers="TEST",
        n_folds=n_folds,
    )

    assert [fold["test_rows"] for fold in fold_results] == [15, 15, 18]
    assert sum(fold["test_rows"] for fold in fold_results) + fold_results[0][
        "train_rows"
    ] == len(data)
    assert fold_results[-1]["oos_results"]["Date"][-1] == dates[-1]

    final_fold = get_validation_data(data, walk_forward=True, n_folds=n_folds)
    assert len(final_fold) == 18
    assert final_fold["Date"][0] == dates[45]
    assert final_fold["Date"][-1] == dates[-1]

    oos_data = get_walk_forward_oos_data(data, n_folds=n_folds)
    assert len(oos_data) == 48
    assert oos_data["Date"][0] == dates[15]
    assert oos_data["Date"][-1] == dates[-1]


def test_stitch_walk_forward_oos_results_rejects_empty_inputs() -> None:
    """Verify stitching rejects empty fold lists or empty fold frames."""
    with pytest.raises(ValueError, match="must not be empty"):
        stitch_walk_forward_oos_results([])

    with pytest.raises(ValueError, match="non-empty oos_results"):
        stitch_walk_forward_oos_results([{"fold": 1, "oos_results": pl.DataFrame()}])


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


def test_get_training_data_standard_split():
    """Verify standard optimisation training uses the 70% train split."""
    data = pl.DataFrame({"Close": list(range(100))})
    training_data = get_training_data(data)
    assert len(training_data) == 70
    assert training_data["Close"][0] == 0
    assert training_data["Close"][-1] == 69


def test_get_training_data_walk_forward():
    """Verify walk-forward training uses the expanding context before final fold."""
    data = pl.DataFrame({"Close": list(range(60))})
    training_data = get_training_data(data, walk_forward=True, n_folds=5)
    assert len(training_data) == 50
    assert training_data["Close"][0] == 0
    assert training_data["Close"][-1] == 49


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


def test_evaluate_validation_period_preserves_rolling_context() -> None:
    """Verify validation evaluation carries rolling state across its boundary."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(12)]
    data = pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0 + i for i in range(12)],
        }
    )
    validation_data = data[8:]
    params = {"short_window": 2, "long_window": 3}

    contextual_results, contextual_metrics = evaluate_validation_period(
        data,
        validation_data,
        "Moving Average Crossover",
        params,
        "TEST",
    )
    isolated_results, isolated_metrics = run_backtest(
        validation_data,
        "Moving Average Crossover",
        params,
        "TEST",
    )

    assert contextual_results["Date"].to_list() == dates[8:]
    assert contextual_results["signal"].to_list() == [1.0] * 4
    assert contextual_results["position_change"].to_list() == [0.0] * 4
    assert contextual_results["cumulative_returns"][0] == pytest.approx(1.0 + 1 / 107)
    expected_cumulative_returns = (
        1 + contextual_results["strategy_returns"]
    ).cum_prod()
    assert contextual_results["cumulative_returns"].to_list() == pytest.approx(
        expected_cumulative_returns.to_list()
    )
    assert contextual_results["equity_curve"].to_list() == pytest.approx(
        (100_000 * expected_cumulative_returns).to_list()
    )
    assert contextual_results["cumulative_transaction_costs"].to_list() == (
        contextual_results["transaction_costs"].cum_sum().to_list()
    )
    assert contextual_metrics["Total Return"] > isolated_metrics["Total Return"]
    assert isolated_results["signal"].to_list()[:2] == [0.0, 0.0]


def test_evaluate_validation_period_uses_full_row_count_for_calmar() -> None:
    """Verify validation Calmar counts day 0 when prior context exists."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 115.0, 105.0, 112.0, 120.0]
    data = pl.DataFrame({"Date": dates, "Close": closes})
    validation_data = data[6:]

    results, metrics = evaluate_validation_period(
        data,
        validation_data,
        "Buy and Hold",
        {},
        "TEST",
    )

    total_return = metrics["Total Return"]
    max_drawdown = abs(metrics["Max Drawdown"])
    assert len(results) == 4
    expected_calmar = (
        (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(results)) - 1
    ) / max_drawdown
    assert metrics["Calmar Ratio"] == pytest.approx(expected_calmar)


def test_evaluate_validation_period_cold_start_uses_n_minus_one_for_calmar() -> None:
    """Verify same-start validation falls back to n - 1 Calmar periods."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
    closes = [100.0, 105.0, 95.0, 102.0, 110.0]
    data = pl.DataFrame({"Date": dates, "Close": closes})

    results, metrics = evaluate_validation_period(
        data,
        data,
        "Buy and Hold",
        {},
        "TEST",
    )

    total_return = metrics["Total Return"]
    max_drawdown = abs(metrics["Max Drawdown"])
    assert len(results) == 5
    expected_calmar = (
        (1 + total_return) ** (TRADING_DAYS_PER_YEAR / (len(results) - 1)) - 1
    ) / max_drawdown
    assert metrics["Calmar Ratio"] == pytest.approx(expected_calmar)


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

    call_log: list[tuple[str, int] | tuple[str, int, int]] = []

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
        call_log.append(("backtest", len(data)))
        return None, {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    def mock_evaluate_validation_period(
        context_data,
        validation_data,
        strategy_type,
        params,
        tickers,
        **_kwargs,
    ):
        call_log.append(("validation", len(context_data), len(validation_data)))
        return pl.DataFrame(), {
            "Sharpe Ratio": 0.5,
            "Total Return": 0.05,
            "Max Drawdown": -0.03,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        mock_evaluate_validation_period,
    )

    _params, metrics = optimise_strategy_params(
        train_data,
        "Moving Average Crossover",
        {"short_window": [5], "long_window": [20]},
        "AAPL",
        test_data=test_data,
    )

    # Should have run backtests for each param combo on train + once on test
    assert call_log == [
        ("backtest", 21),
        ("validation", 31, 10),
    ]
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

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
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

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
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

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
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

    def mock_backtest_init(data, strategy, tickers=None, **_kwargs):
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


def test_importing_optimiser_does_not_import_streamlit_or_warn() -> None:
    """Verify importing optimiser does not load streamlit or emit warnings."""
    isolated_script = (
        "import logging, sys, types, warnings\n"
        "log_records = []\n"
        "class CaptureHandler(logging.Handler):\n"
        "    def emit(self, record):\n"
        "        if record.levelno >= logging.WARNING:\n"
        "            log_records.append(record.getMessage())\n"
        "logging.getLogger().addHandler(CaptureHandler())\n"
        "stub = types.ModuleType('quant_trading_strategy_backtester.backtester')\n"
        "stub.Backtester = object\n"
        "sys.modules['quant_trading_strategy_backtester.backtester'] = stub\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import quant_trading_strategy_backtester.optimiser\n"
        "assert 'streamlit' not in sys.modules, 'streamlit was imported'\n"
        "assert not caught, f'Unexpected warnings: {caught}'\n"
        "assert not log_records, f'Unexpected log warnings: {log_records}'\n"
    )
    isolated_result = subprocess.run(
        [sys.executable, "-c", isolated_script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert isolated_result.returncode == 0, isolated_result.stderr
    assert "No runtime found" not in isolated_result.stderr
    assert "streamlit" not in isolated_result.stderr.lower()

    direct_script = (
        "import logging, warnings\n"
        "log_records = []\n"
        "class CaptureHandler(logging.Handler):\n"
        "    def emit(self, record):\n"
        "        if record.levelno >= logging.WARNING:\n"
        "            log_records.append(record.getMessage())\n"
        "logging.getLogger().addHandler(CaptureHandler())\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import quant_trading_strategy_backtester.optimiser\n"
        "assert not caught, f'Unexpected warnings: {caught}'\n"
        "assert not log_records, f'Unexpected log warnings: {log_records}'\n"
    )
    direct_result = subprocess.run(
        [sys.executable, "-c", direct_script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert direct_result.returncode == 0, direct_result.stderr
    assert "No runtime found" not in direct_result.stderr


def test_optimise_strategy_params_invokes_callbacks(monkeypatch) -> None:
    """Verify parameter optimisation invokes progress and status callbacks."""
    data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 22)],
            "Close": [100.0 + i for i in range(21)],
        }
    )

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
        return None, {
            "Sharpe Ratio": float(params["short_window"]),
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest", mock_run_backtest
    )

    progress_events: list[tuple[int, int, str]] = []
    status_messages: list[str] = []

    optimise_strategy_params(
        data,
        "Moving Average Crossover",
        {"short_window": [5, 10, 15], "long_window": [20]},
        "AAPL",
        progress_callback=lambda current, total, msg: progress_events.append(
            (current, total, msg)
        ),
        status_callback=status_messages.append,
    )

    assert progress_events == [
        (1, 3, "Evaluating parameter combination 1 / 3"),
        (2, 3, "Evaluating parameter combination 2 / 3"),
        (3, 3, "Evaluating parameter combination 3 / 3"),
    ]
    assert len(status_messages) == 1
    assert "3 valid combinations" in status_messages[0]
    assert "21 training rows" in status_messages[0]


def test_optimise_buy_and_hold_ticker_invokes_callbacks(monkeypatch) -> None:
    """Verify buy-and-hold ticker selection reports progress and status."""
    top_companies = [("AAPL", 2.0), ("MSFT", 1.0)]
    mock_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100.0 + i for i in range(31)],
        }
    )

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: mock_data,
    )

    progress_events: list[tuple[int, int, str]] = []
    status_messages: list[str] = []

    optimise_buy_and_hold_ticker(
        top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 31),
        progress_callback=lambda current, total, msg: progress_events.append(
            (current, total, msg)
        ),
        status_callback=status_messages.append,
    )

    assert progress_events == [
        (1, 2, "Evaluating ticker 1 / 2: AAPL"),
        (2, 2, "Evaluating ticker 2 / 2: MSFT"),
    ]
    assert len(status_messages) == 1
    assert "evaluated 2 of 2 candidates" in status_messages[0]


def test_optimise_single_ticker_strategy_ticker_invokes_callbacks(
    monkeypatch,
) -> None:
    """Verify single-ticker selection reports progress and status."""
    top_companies = [("AAPL", 2.0), ("GOOGL", 1.5), ("MSFT", 1.0)]
    mock_data = pl.DataFrame(
        {
            "Date": [datetime.date(2020, 1, i) for i in range(1, 32)],
            "Close": [100.0 + i for i in range(31)],
        }
    )

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: mock_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        lambda *_args, **_kwargs: (None, {"Sharpe Ratio": 1.2}),
    )

    progress_events: list[tuple[int, int, str]] = []
    status_messages: list[str] = []

    optimise_single_ticker_strategy_ticker(
        top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 31),
        "Mean Reversion",
        {"window": 10, "std_dev": 2.0},
        progress_callback=lambda current, total, msg: progress_events.append(
            (current, total, msg)
        ),
        status_callback=status_messages.append,
    )

    assert progress_events == [
        (1, 3, "Evaluating ticker 1 / 3: AAPL"),
        (2, 3, "Evaluating ticker 2 / 3: GOOGL"),
        (3, 3, "Evaluating ticker 3 / 3: MSFT"),
    ]
    assert len(status_messages) == 1
    assert "evaluated 3 of 3 candidates" in status_messages[0]


def test_optimise_pairs_trading_tickers_invokes_callbacks(monkeypatch) -> None:
    """Verify pairs ticker selection reports progress and status messages."""
    top_companies = [("AAPL", 3.0), ("GOOGL", 2.0), ("MSFT", 1.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(20)]
    mock_batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i * 0.1 for i in range(20)],
            "GOOGL": [200.0 + i * 0.2 for i in range(20)],
            "MSFT": [300.0 + i * 0.3 for i in range(20)],
        }
    )

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        lambda *_args, **_kwargs: mock_batch_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        lambda *_args, **_kwargs: (None, {"Sharpe Ratio": 1.5}),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        lambda *_args, **_kwargs: (pl.DataFrame(), {"Sharpe Ratio": 1.1}),
    )

    progress_events: list[tuple[int, int, str]] = []
    status_messages: list[str] = []

    optimise_pairs_trading_tickers(
        top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 20),
        {"window": [10], "entry_z_score": [2.0], "exit_z_score": [0.5]},
        optimise=True,
        progress_callback=lambda current, total, msg: progress_events.append(
            (current, total, msg)
        ),
        status_callback=status_messages.append,
    )

    assert [(c, t) for c, t, _ in progress_events] == [(1, 3), (2, 3), (3, 3)]
    assert "AAPL vs. GOOGL" in progress_events[0][2]
    assert "AAPL vs. MSFT" in progress_events[1][2]
    assert "GOOGL vs. MSFT" in progress_events[2][2]
    assert len(status_messages) == 2
    assert "evaluated 3 of 3 candidate pairs" in status_messages[0]
    assert "rejected 0" in status_messages[0]
    assert "1 valid combinations per cointegrated pair" in status_messages[1]


def test_run_optimisation_walk_forward_invokes_callbacks() -> None:
    """Verify run_optimisation forwards progress, status, and fold hooks."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(60)]
    data = pl.DataFrame({"Date": dates, "Close": [100.0 + i * 0.5 for i in range(60)]})

    progress_events: list[tuple[int, int, str]] = []
    status_messages: list[str] = []
    received_folds: list[list[dict[str, Any]]] = []

    run_optimisation(
        data,
        "Moving Average Crossover",
        {"short_window": [3, 5], "long_window": [10]},
        datetime.date(2020, 1, 1),
        datetime.date(2020, 3, 1),
        "TEST",
        walk_forward=True,
        progress_callback=lambda current, total, msg: progress_events.append(
            (current, total, msg)
        ),
        status_callback=status_messages.append,
        fold_results_callback=received_folds.append,
    )

    assert len(progress_events) == 10
    assert progress_events[0][:2] == (1, 10)
    assert progress_events[-1][:2] == (10, 10)
    assert len(status_messages) == 1
    assert (
        "2 valid combinations across 5 expanding walk-forward folds"
        in (status_messages[0])
    )
    assert len(received_folds) == 1
    assert len(received_folds[0]) == 5


def test_optimise_single_ticker_picks_first_valid_parameter_combination(
    monkeypatch,
) -> None:
    """Verify ticker selection skips invalid leading parameter combinations."""
    top_companies = [("AAPL", 2.0), ("MSFT", 1.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(40)]
    mock_data = pl.DataFrame(
        {"Date": dates, "Close": [100.0 + i * 0.5 for i in range(40)]}
    )
    used_params: list[dict[str, Any]] = []

    def tracking_run_backtest(
        data,
        strategy_type,
        params,
        tickers,
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
    ):
        used_params.append(
            {
                "params": params,
                "initial_capital": initial_capital,
                "transaction_cost_bps": transaction_cost_bps,
                "slippage_bps": slippage_bps,
            }
        )
        return None, {"Sharpe Ratio": 1.0 if tickers == "AAPL" else 2.0}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_one_ticker",
        lambda *_args, **_kwargs: mock_data,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        tracking_run_backtest,
    )

    best_ticker = optimise_single_ticker_strategy_ticker(
        top_companies,
        dates[0],
        dates[-1],
        "Moving Average Crossover",
        {"short_window": [25, 10], "long_window": [20, 50]},
        initial_capital=250_000.0,
        transaction_cost_bps=12.0,
        slippage_bps=6.0,
    )

    assert best_ticker == "MSFT"
    # (25, 20) fails short_window < long_window; (25, 50) is first valid.
    assert used_params == [
        {
            "params": {"short_window": 25, "long_window": 50},
            "initial_capital": 250_000.0,
            "transaction_cost_bps": 12.0,
            "slippage_bps": 6.0,
        },
        {
            "params": {"short_window": 25, "long_window": 50},
            "initial_capital": 250_000.0,
            "transaction_cost_bps": 12.0,
            "slippage_bps": 6.0,
        },
    ]


def test_optimise_single_ticker_raises_when_no_valid_parameter_combination() -> None:
    """Verify ticker selection fails when no parameter combo is valid."""
    with pytest.raises(ValueError, match="No valid parameter combinations"):
        optimise_single_ticker_strategy_ticker(
            [("AAPL", 2.0)],
            datetime.date(2020, 1, 1),
            datetime.date(2020, 3, 1),
            "Moving Average Crossover",
            {"short_window": [30, 40], "long_window": [20]},
        )


def test_trading_assumptions_propagate_through_optimiser() -> None:
    """Verify custom capital and cost assumptions flow through optimiser."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(72)]
    closes = [100.0 + (i % 10) * 1.5 + i * 0.2 for i in range(72)]
    data = pl.DataFrame({"Date": dates, "Close": closes})

    low_cost_results, low_cost_metrics = run_backtest(
        data,
        "Moving Average Crossover",
        {"short_window": 3, "long_window": 8},
        "AAPL",
        initial_capital=50_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    high_cost_results, high_cost_metrics = run_backtest(
        data,
        "Moving Average Crossover",
        {"short_window": 3, "long_window": 8},
        "AAPL",
        initial_capital=250_000.0,
        transaction_cost_bps=50.0,
        slippage_bps=25.0,
    )

    assert low_cost_results["equity_curve"][0] == pytest.approx(50_000.0)
    assert high_cost_results["equity_curve"][0] == pytest.approx(250_000.0)
    assert high_cost_metrics["Total Costs"] > low_cost_metrics["Total Costs"]
    assert high_cost_metrics["Total Return"] < low_cost_metrics["Total Return"]

    val_results, val_metrics = evaluate_validation_period(
        data,
        data[50:],
        "Moving Average Crossover",
        {"short_window": 3, "long_window": 8},
        "AAPL",
        initial_capital=200_000.0,
        transaction_cost_bps=50.0,
        slippage_bps=25.0,
    )
    val_zero_results, val_zero_metrics = evaluate_validation_period(
        data,
        data[50:],
        "Moving Average Crossover",
        {"short_window": 3, "long_window": 8},
        "AAPL",
        initial_capital=200_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    assert val_zero_results["equity_curve"][0] == pytest.approx(
        200_000.0 * (1 + val_zero_results["strategy_returns"][0])
    )
    assert val_metrics["Total Costs"] > val_zero_metrics["Total Costs"]
    assert val_results["equity_curve"][-1] < val_zero_results["equity_curve"][-1]

    _, wf_high_metrics, wf_high_folds = walk_forward_optimise(
        data,
        "Moving Average Crossover",
        {"short_window": [3], "long_window": [8]},
        "AAPL",
        n_folds=3,
        initial_capital=300_000.0,
        transaction_cost_bps=50.0,
        slippage_bps=25.0,
    )
    _, wf_zero_metrics, wf_zero_folds = walk_forward_optimise(
        data,
        "Moving Average Crossover",
        {"short_window": [3], "long_window": [8]},
        "AAPL",
        n_folds=3,
        initial_capital=300_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    wf_stitched, _ = stitch_walk_forward_oos_results(
        wf_high_folds,
        initial_capital=300_000.0,
        transaction_cost_bps=50.0,
        slippage_bps=25.0,
    )
    assert wf_stitched["equity_curve"][0] == pytest.approx(
        300_000.0 * (1 + wf_stitched["strategy_returns"][0])
    )
    assert wf_high_metrics["Total Costs"] > wf_zero_metrics["Total Costs"]
    assert len(wf_high_folds) == len(wf_zero_folds) == 3


def test_optimise_buy_and_hold_and_pairs_forward_trading_assumptions(
    monkeypatch,
) -> None:
    """Verify buy-and-hold and pairs ticker selection forward assumptions."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(30)]
    single_data = pl.DataFrame(
        {"Date": dates, "Close": [100.0 + i * 0.5 for i in range(30)]}
    )
    pair_data = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": [100.0 + i * 0.5 for i in range(30)],
            "Close_2": [200.0 + i * 1.0 for i in range(30)],
        }
    )

    _, _, bh_zero = optimise_buy_and_hold_ticker(
        [("AAPL", 1.0)],
        dates[0],
        dates[-1],
        load_one_ticker=lambda *_a: single_data,
        initial_capital=150_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    _, _, bh_high = optimise_buy_and_hold_ticker(
        [("AAPL", 1.0)],
        dates[0],
        dates[-1],
        load_one_ticker=lambda *_a: single_data,
        initial_capital=150_000.0,
        transaction_cost_bps=40.0,
        slippage_bps=20.0,
    )
    assert bh_high["Total Costs"] > bh_zero["Total Costs"]
    assert bh_high["Total Return"] < bh_zero["Total Return"]

    captured_pair_kwargs: list[dict[str, float]] = []

    def tracking_run_backtest(
        data,
        strategy_type,
        params,
        tickers,
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
    ):
        captured_pair_kwargs.append(
            {
                "initial_capital": initial_capital,
                "transaction_cost_bps": transaction_cost_bps,
                "slippage_bps": slippage_bps,
            }
        )
        return None, {"Sharpe Ratio": 1.5}

    def tracking_eval(
        context_data,
        validation_data,
        strategy_type,
        params,
        tickers,
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
    ):
        captured_pair_kwargs.append(
            {
                "initial_capital": initial_capital,
                "transaction_cost_bps": transaction_cost_bps,
                "slippage_bps": slippage_bps,
            }
        )
        return pl.DataFrame(), {"Sharpe Ratio": 1.2}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_a, **_k: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        tracking_run_backtest,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        tracking_eval,
    )

    optimise_pairs_trading_tickers(
        [("AAPL", 2.0), ("MSFT", 1.0)],
        dates[0],
        dates[-1],
        {"window": 10, "entry_z_score": 2.0, "exit_z_score": 0.5},
        optimise=False,
        load_two_tickers=lambda *_a: pair_data,
        same_company_check=lambda *_a: False,
        initial_capital=175_000.0,
        transaction_cost_bps=15.0,
        slippage_bps=7.5,
    )

    assert captured_pair_kwargs == [
        {
            "initial_capital": 175_000.0,
            "transaction_cost_bps": 15.0,
            "slippage_bps": 7.5,
        },
        {
            "initial_capital": 175_000.0,
            "transaction_cost_bps": 15.0,
            "slippage_bps": 7.5,
        },
    ]


def test_count_candidate_pairs_uses_company_ciks_without_yfinance_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CIK filtering excludes dual-class pairs with zero Yahoo calls."""
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda *_args, **_kwargs: pytest.fail("Yahoo Finance should not be called"),
    )
    top_companies = [
        ("GOOG", 5.0),
        ("GOOGL", 4.5),
        ("FOX", 3.0),
        ("FOXA", 2.5),
        ("MSFT", 1.0),
    ]
    ciks = {
        "GOOG": "0001652044",
        "GOOGL": "0001652044",
        "FOX": "0001754301",
        "FOXA": "0001754301",
        "MSFT": "0000789019",
    }

    # C(5, 2) = 10 total pairs minus (GOOG, GOOGL) and (FOX, FOXA) = 8.
    assert count_candidate_pairs(top_companies, company_ciks=ciks) == 8


def test_count_candidate_pairs_fallback_bounds_yfinance_calls_to_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback filtering makes at most N Yahoo .info lookups."""
    names = {
        "GOOG": "Alphabet Inc.",
        "GOOGL": "Alphabet Inc.",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA Corporation",
    }
    info_calls: list[str] = []

    def mock_ticker(symbol: str):
        class MockTicker:
            @property
            def info(self) -> dict[str, str]:
                info_calls.append(symbol)
                return {"longName": names[symbol]}

        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker)
    top_companies = [(ticker, float(idx + 1)) for idx, ticker in enumerate(names)]

    # C(5, 2) = 10 pairs minus (GOOG, GOOGL) = 9, with <= 5 Yahoo calls.
    assert count_candidate_pairs(top_companies) == 9
    assert len(info_calls) == 5


def test_optimise_pairs_trading_tickers_batch_downloads_once_and_joins_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pair selection batch-downloads once and drops nulls per pair."""
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda *_args, **_kwargs: pytest.fail("Yahoo Finance should not be called"),
    )
    top_companies = [
        ("AAPL", 4.0),
        ("MSFT", 3.0),
        ("GOOG", 2.0),
        ("GOOGL", 1.5),
    ]
    ciks = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOG": "0001652044",
        "GOOGL": "0001652044",
    }
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    batch_data = pl.DataFrame(
        {
            "Date": dates,
            "AAPL": [100.0 + i for i in range(10)],
            "MSFT": [None if i == 0 else 200.0 + i for i in range(10)],
            "GOOG": [300.0 + i for i in range(10)],
            "GOOGL": [310.0 + i for i in range(10)],
        }
    )

    batch_calls: list[tuple[str, ...]] = []
    seen_train_lengths: dict[tuple[str, str], int] = {}

    def mock_load_many(tickers, start_date, end_date):
        batch_calls.append(tuple(tickers))
        return batch_data

    def mock_run_backtest(data, strategy_type, params, tickers, **_kwargs):
        pair = tuple(tickers)
        seen_train_lengths[pair] = len(data)
        sharpe = 2.5 if pair == ("AAPL", "GOOG") else 1.0
        return None, {
            "Sharpe Ratio": sharpe,
            "Total Return": 0.1,
            "Max Drawdown": -0.05,
        }

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        mock_run_backtest,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        lambda *_args, **_kwargs: (pl.DataFrame(), {"Sharpe Ratio": 2.0}),
    )

    best_pair, _, metrics = optimise_pairs_trading_tickers(
        top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 11),
        {"window": 5, "entry_z_score": 2.0, "exit_z_score": 0.5},
        optimise=False,
        load_many_tickers=mock_load_many,
        company_ciks=ciks,
    )

    # Single batch call for unique tickers, and (GOOG, GOOGL) excluded by CIK.
    assert batch_calls == [("AAPL", "MSFT", "GOOG", "GOOGL")]
    assert ("GOOG", "GOOGL") not in seen_train_lengths
    # AAPL/GOOG retains all 10 rows (7 train); AAPL/MSFT drops day 0 (6 train).
    assert seen_train_lengths[("AAPL", "GOOG")] == 7
    assert seen_train_lengths[("AAPL", "MSFT")] == 6
    assert best_pair == ("AAPL", "GOOG")
    assert metrics["Sharpe Ratio"] == 2.0


def test_optimise_pairs_trading_tickers_honours_load_two_tickers_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit load_two_tickers override bypasses batch downloads."""
    top_companies = [("AAPL", 3.0), ("MSFT", 2.0), ("GOOGL", 1.0)]
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    pair_frame = pl.DataFrame(
        {
            "Date": dates,
            "Close_1": [100.0 + i for i in range(10)],
            "Close_2": [200.0 + i for i in range(10)],
        }
    )
    pair_calls: list[tuple[str, str]] = []

    def custom_pair_loader(ticker1, ticker2, start_date, end_date):
        pair_calls.append((ticker1, ticker2))
        return pair_frame

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.load_yfinance_data_many_tickers",
        lambda *_args, **_kwargs: pytest.fail("Batch loader must not be called"),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.is_same_company",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_cointegration",
        lambda *_args, **_kwargs: _cointegration_result(True),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        lambda *_args, **_kwargs: (None, {"Sharpe Ratio": 1.2}),
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        lambda *_args, **_kwargs: (pl.DataFrame(), {"Sharpe Ratio": 1.0}),
    )

    optimise_pairs_trading_tickers(
        top_companies,
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 11),
        {"window": 5, "entry_z_score": 2.0, "exit_z_score": 0.5},
        optimise=False,
        load_two_tickers=custom_pair_loader,
    )

    assert pair_calls == [("AAPL", "MSFT"), ("AAPL", "GOOGL"), ("MSFT", "GOOGL")]


def _make_pairs_transition_data(n_rows: int = 36) -> pl.DataFrame:
    """Build deterministic cointegrated pair prices with spread excursions."""
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    # Use oscillating Close_2 so rolling variance is always strictly positive
    # and rolling OLS beta remains positive and well-conditioned.
    close_2 = [100.0 + 2.0 * (i % 4) + 0.4 * i for i in range(n_rows)]
    # Add spread deviations that trigger long and short entries across windows.
    spread_offsets = [
        0.2,
        -0.2,
        0.1,
        -0.1,
        0.2,
        -0.1,
        0.1,
        -0.2,
        0.15,
        -0.15,
        0.1,
        -0.1,
        0.05,
        -0.05,
        -4.0,
        -6.0,
        -8.0,
        -10.0,
        -0.5,
        0.5,
        5.0,
        7.0,
        9.0,
        0.0,
    ]
    close_1 = [
        1.5 * close_2[i] + spread_offsets[i % len(spread_offsets)]
        for i in range(n_rows)
    ]
    return pl.DataFrame({"Date": dates, "Close_1": close_1, "Close_2": close_2})


def test_evaluate_validation_period_reconciles_single_asset_fold_transition() -> None:
    """Verify single-asset fold transitions earn prior return and pay costs."""
    closes = [
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        110.0,
        120.0,
        130.0,
        120.0,
        115.0,
        117.0,
        110.0,
    ]
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
        for i in range(len(closes))
    ]
    data = pl.DataFrame({"Date": dates, "Close": closes})
    tx_bps = 20.0
    slip_bps = 10.0
    cost_rate = (tx_bps + slip_bps) / 10_000

    # Fold 1 (bars 6..9) uses window=3, which enters LONG (+1.0) on bar 9.
    fold_1_params = {"window": 3, "std_dev": 0.5}
    fold_1_results, _ = evaluate_validation_period(
        data[:10],
        data[6:10],
        "Mean Reversion",
        fold_1_params,
        "TEST",
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    assert fold_1_results["signal"][-1] == pytest.approx(1.0)

    # Fold 2 (bars 10..11) uses window=8, which was already SHORT (-1.0) on
    # bar 9 and remains SHORT (-1.0) on bar 10.
    fold_2_params = {"window": 8, "std_dev": 0.5}
    unreconciled_results, _ = evaluate_validation_period(
        data,
        data[10:],
        "Mean Reversion",
        fold_2_params,
        "TEST",
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    assert unreconciled_results["signal"][0] == pytest.approx(-1.0)
    assert unreconciled_results["position_change"][0] == pytest.approx(0.0)
    assert unreconciled_results["trade_turnover"][0] == pytest.approx(0.0)

    prior_position = PortfolioPosition(signal=float(fold_1_results["signal"][-1]))
    fold_2_results, fold_2_metrics = evaluate_validation_period(
        data,
        data[10:],
        "Mean Reversion",
        fold_2_params,
        "TEST",
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
        prior_position=prior_position,
    )

    expected_asset_ret = (117.0 - 115.0) / 115.0
    expected_gross_ret = 1.0 * expected_asset_ret
    expected_cost = 2.0 * cost_rate
    expected_net_ret = expected_gross_ret - expected_cost

    assert fold_2_results["gross_strategy_returns"][0] == pytest.approx(
        expected_gross_ret
    )
    assert fold_2_results["position_change"][0] == pytest.approx(-2.0)
    assert fold_2_results["trade_turnover"][0] == pytest.approx(2.0)
    assert fold_2_results["transaction_costs"][0] == pytest.approx(expected_cost)
    assert fold_2_results["strategy_returns"][0] == pytest.approx(expected_net_ret)
    assert fold_2_results["cumulative_returns"][0] == pytest.approx(
        1.0 + expected_net_ret
    )
    assert fold_2_results["equity_curve"][0] == pytest.approx(
        100_000.0 * (1.0 + expected_net_ret)
    )
    assert fold_2_metrics["Total Costs"] == pytest.approx(
        float(fold_2_results["transaction_costs"].sum())
    )
    assert fold_2_metrics["Total Return"] == pytest.approx(
        float(fold_2_results["cumulative_returns"][-1]) - 1.0
    )

    # Stitching and trade ledger must capture the boundary flip accurately.
    stitched_results, _ = stitch_walk_forward_oos_results(
        [
            {"fold": 1, "oos_results": fold_1_results},
            {"fold": 2, "oos_results": fold_2_results},
        ],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    ledger = build_trade_ledger(stitched_results)
    boundary_rows = ledger.filter(pl.col("Date") == dates[10])
    assert boundary_rows.height == 1
    assert boundary_rows["Action"][0] == "Flip Long To Short"
    assert boundary_rows["Previous Signal"][0] == pytest.approx(1.0)
    assert boundary_rows["Signal"][0] == pytest.approx(-1.0)
    assert boundary_rows["Position Change"][0] == pytest.approx(-2.0)
    assert boundary_rows["Turnover"][0] == pytest.approx(2.0)
    assert boundary_rows["Gross Return"][0] == pytest.approx(expected_gross_ret)
    assert boundary_rows["Transaction Costs"][0] == pytest.approx(expected_cost)
    assert boundary_rows["Net Return"][0] == pytest.approx(expected_net_ret)


def test_evaluate_validation_period_reconciles_pairs_fold_transition() -> None:
    """Verify pairs fold transitions carry leg weights and charge costs."""
    data = _make_pairs_transition_data(24)
    tx_bps = 15.0
    slip_bps = 5.0
    cost_rate = (tx_bps + slip_bps) / 10_000

    # Fold 1 (bars 12..19) ends SHORT (-1.0) with window=5 entry-locked weights.
    fold_1_params = {"window": 5, "entry_z_score": 1.0, "exit_z_score": 0.2}
    fold_1_results, _ = evaluate_validation_period(
        data[:20],
        data[12:20],
        "Pairs Trading",
        fold_1_params,
        ["A", "B"],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    assert fold_1_results["signal"][-1] == pytest.approx(-1.0)
    prior_w1 = float(fold_1_results["leg_1_weight"][-1])
    prior_w2 = float(fold_1_results["leg_2_weight"][-1])
    assert prior_w1 < 0.0
    assert prior_w2 > 0.0

    # Fold 2 (bars 20..23) uses window=7, which is also SHORT (-1.0) on bar 20
    # but with different entry-locked leg weights (and zero unreconciled
    # turnover because window=7 was already SHORT on bar 19).
    fold_2_params = {"window": 7, "entry_z_score": 1.0, "exit_z_score": 0.2}
    unreconciled_results, _ = evaluate_validation_period(
        data,
        data[20:],
        "Pairs Trading",
        fold_2_params,
        ["A", "B"],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    assert unreconciled_results["trade_turnover"][0] == pytest.approx(0.0)

    prior_pos = PortfolioPosition(
        signal=float(fold_1_results["signal"][-1]),
        leg_1_weight=prior_w1,
        leg_2_weight=prior_w2,
    )
    fold_2_results, fold_2_metrics = evaluate_validation_period(
        data,
        data[20:],
        "Pairs Trading",
        fold_2_params,
        ["A", "B"],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
        prior_position=prior_pos,
    )

    target_sig = float(fold_2_results["signal"][0])
    target_w1 = float(fold_2_results["leg_1_weight"][0])
    target_w2 = float(fold_2_results["leg_2_weight"][0])
    assert target_sig == pytest.approx(-1.0)
    assert not math.isclose(target_w1, prior_w1, rel_tol=1e-9, abs_tol=1e-9)

    a1_ret = (data["Close_1"][20] - data["Close_1"][19]) / data["Close_1"][19]
    a2_ret = (data["Close_2"][20] - data["Close_2"][19]) / data["Close_2"][19]
    expected_gross = prior_w1 * a1_ret + prior_w2 * a2_ret
    expected_l1_change = target_w1 - prior_w1
    expected_l2_change = target_w2 - prior_w2
    expected_turnover = abs(expected_l1_change) + abs(expected_l2_change)
    expected_cost = expected_turnover * cost_rate
    expected_net = expected_gross - expected_cost

    assert fold_2_results["gross_strategy_returns"][0] == pytest.approx(expected_gross)
    assert fold_2_results["position_change"][0] == pytest.approx(0.0)
    assert fold_2_results["leg_1_weight_change"][0] == pytest.approx(expected_l1_change)
    assert fold_2_results["leg_2_weight_change"][0] == pytest.approx(expected_l2_change)
    assert fold_2_results["trade_turnover"][0] == pytest.approx(expected_turnover)
    assert fold_2_results["pair_turnover"][0] == pytest.approx(expected_turnover)
    assert fold_2_results["transaction_costs"][0] == pytest.approx(expected_cost)
    assert fold_2_results["strategy_returns"][0] == pytest.approx(expected_net)
    assert fold_2_metrics["Total Costs"] == pytest.approx(
        float(fold_2_results["transaction_costs"].sum())
    )

    # Verify stitched trade ledger records the leg-weight rebalance.
    stitched_rebalance, _ = stitch_walk_forward_oos_results(
        [
            {"fold": 1, "params": fold_1_params, "oos_results": fold_1_results},
            {"fold": 2, "params": fold_2_params, "oos_results": fold_2_results},
        ],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    rebalance_ledger = build_trade_ledger(stitched_rebalance)
    rebalance_rows = rebalance_ledger.filter(pl.col("Date") == data["Date"][20])
    assert rebalance_rows.height == 1
    assert rebalance_rows["Action"][0] == "Rebalance"
    assert rebalance_rows["Turnover"][0] == pytest.approx(expected_turnover)
    assert rebalance_rows["Transaction Costs"][0] == pytest.approx(expected_cost)

    # Verify a signal flip across the boundary (LONG prior -> SHORT target).
    flip_prior = PortfolioPosition(signal=1.0, leg_1_weight=0.6, leg_2_weight=-0.4)
    flip_results, _ = evaluate_validation_period(
        data,
        data[20:],
        "Pairs Trading",
        fold_2_params,
        ["A", "B"],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
        prior_position=flip_prior,
    )
    flip_gross = 0.6 * a1_ret - 0.4 * a2_ret
    flip_l1_change = target_w1 - 0.6
    flip_l2_change = target_w2 - (-0.4)
    flip_turnover = abs(flip_l1_change) + abs(flip_l2_change)
    flip_cost = flip_turnover * cost_rate

    assert flip_results["gross_strategy_returns"][0] == pytest.approx(flip_gross)
    assert flip_results["position_change"][0] == pytest.approx(-2.0)
    assert flip_results["leg_1_weight_change"][0] == pytest.approx(flip_l1_change)
    assert flip_results["leg_2_weight_change"][0] == pytest.approx(flip_l2_change)
    assert flip_results["trade_turnover"][0] == pytest.approx(flip_turnover)
    assert flip_results["pair_turnover"][0] == pytest.approx(flip_turnover)
    assert flip_results["transaction_costs"][0] == pytest.approx(flip_cost)
    assert flip_results["strategy_returns"][0] == pytest.approx(flip_gross - flip_cost)


def test_walk_forward_identical_params_matches_continuous_backtest() -> None:
    """Verify identical fold params produce zero spurious transition costs."""
    tx_bps = 25.0
    slip_bps = 10.0

    # Single-asset continuity check with oscillating prices and active trades.
    n_rows = 48
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)
    ]
    closes = [100.0 + 8.0 * math.sin(i * 0.7) + 0.1 * i for i in range(n_rows)]
    single_data = pl.DataFrame({"Date": dates, "Close": closes})
    single_params = {"window": 4, "std_dev": 0.8}

    _, wf_single_metrics, wf_single_folds = walk_forward_optimise(
        single_data,
        "Mean Reversion",
        {"window": [4], "std_dev": [0.8]},
        "TEST",
        n_folds=3,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    stitched_single, _ = stitch_walk_forward_oos_results(
        wf_single_folds,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    single_oos_data = get_walk_forward_oos_data(single_data, n_folds=3)
    continuous_single, continuous_single_metrics = evaluate_validation_period(
        single_data,
        single_oos_data,
        "Mean Reversion",
        single_params,
        "TEST",
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )

    for col in (
        "signal",
        "position_change",
        "gross_strategy_returns",
        "trade_turnover",
        "transaction_costs",
        "strategy_returns",
        "equity_curve",
    ):
        assert stitched_single[col].to_list() == pytest.approx(
            continuous_single[col].to_list()
        )
    assert wf_single_metrics == pytest.approx(continuous_single_metrics, nan_ok=True)

    # Pairs-trading continuity check with identical params across folds.
    pairs_data = _make_pairs_transition_data(36)
    pairs_params = {"window": 5, "entry_z_score": 1.0, "exit_z_score": 0.2}

    _, wf_pairs_metrics, wf_pairs_folds = walk_forward_optimise(
        pairs_data,
        "Pairs Trading",
        {"window": [5], "entry_z_score": [1.0], "exit_z_score": [0.2]},
        ["A", "B"],
        n_folds=3,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    stitched_pairs, _ = stitch_walk_forward_oos_results(
        wf_pairs_folds,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    pairs_oos_data = get_walk_forward_oos_data(pairs_data, n_folds=3)
    continuous_pairs, continuous_pairs_metrics = evaluate_validation_period(
        pairs_data,
        pairs_oos_data,
        "Pairs Trading",
        pairs_params,
        ["A", "B"],
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )

    for col in (
        "signal",
        "position_change",
        "leg_1_weight",
        "leg_2_weight",
        "leg_1_weight_change",
        "leg_2_weight_change",
        "gross_strategy_returns",
        "trade_turnover",
        "pair_turnover",
        "transaction_costs",
        "strategy_returns",
        "equity_curve",
    ):
        assert stitched_pairs[col].to_list() == pytest.approx(
            continuous_pairs[col].to_list()
        )
    assert wf_pairs_metrics == pytest.approx(continuous_pairs_metrics, nan_ok=True)


def test_walk_forward_optimise_threads_prior_position_across_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify walk_forward_optimise carries closing positions end-to-end."""
    # 15 bars with n_folds=2 -> segment_size=5:
    # train1=[0..5), test1=[5..10); train2=[0..10), test2=[10..15).
    closes = [
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        110.0,
        120.0,
        130.0,
        120.0,
        115.0,
        117.0,
        110.0,
        108.0,
        106.0,
        105.0,
    ]
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
        for i in range(len(closes))
    ]
    data = pl.DataFrame({"Date": dates, "Close": closes})
    tx_bps = 20.0
    slip_bps = 10.0
    cost_rate = (tx_bps + slip_bps) / 10_000

    original_run_backtest = run_backtest
    original_eval = evaluate_validation_period
    received_prior_positions: list[PortfolioPosition | None] = []

    def guided_run_backtest(
        frame: pl.DataFrame,
        strategy_type: str,
        params: dict[str, Any],
        tickers: str | list[str],
        **kwargs: Any,
    ) -> tuple[pl.DataFrame, dict[str, float]]:
        results, metrics = original_run_backtest(
            frame, strategy_type, params, tickers, **kwargs
        )
        # Steer fold 1 training (len 5) to pick window=3 (ends LONG on bar 9)
        # and fold 2 training (len 10) to pick window=8 (SHORT on bars 9 & 10).
        if len(frame) == 5:
            metrics = dict(metrics)
            metrics["Sharpe Ratio"] = 2.0 if params["window"] == 3 else 1.0
        elif len(frame) == 10:
            metrics = dict(metrics)
            metrics["Sharpe Ratio"] = 2.0 if params["window"] == 8 else 1.0
        return results, metrics

    def spying_eval(*args: Any, **kwargs: Any) -> tuple[pl.DataFrame, dict[str, float]]:
        received_prior_positions.append(kwargs.get("prior_position"))
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        guided_run_backtest,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        spying_eval,
    )

    best_params, stitched_metrics, fold_results = walk_forward_optimise(
        data,
        "Mean Reversion",
        {"window": [3, 8], "std_dev": [0.5]},
        "TEST",
        n_folds=2,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )

    assert best_params == {"window": 8, "std_dev": 0.5}
    assert fold_results[0]["params"] == {"window": 3, "std_dev": 0.5}
    assert fold_results[1]["params"] == {"window": 8, "std_dev": 0.5}
    assert fold_results[0]["oos_results"]["signal"][-1] == pytest.approx(1.0)
    assert fold_results[1]["oos_results"]["signal"][0] == pytest.approx(-1.0)
    assert received_prior_positions == [None, PortfolioPosition(signal=1.0)]

    expected_asset_ret = (117.0 - 115.0) / 115.0
    expected_cost = 2.0 * cost_rate
    fold_2_oos = fold_results[1]["oos_results"]
    assert fold_2_oos["gross_strategy_returns"][0] == pytest.approx(expected_asset_ret)
    assert fold_2_oos["position_change"][0] == pytest.approx(-2.0)
    assert fold_2_oos["trade_turnover"][0] == pytest.approx(2.0)
    assert fold_2_oos["transaction_costs"][0] == pytest.approx(expected_cost)
    assert fold_2_oos["strategy_returns"][0] == pytest.approx(
        expected_asset_ret - expected_cost
    )

    stitched_results, recomputed_metrics = stitch_walk_forward_oos_results(
        fold_results,
        transaction_cost_bps=tx_bps,
        slippage_bps=slip_bps,
    )
    assert stitched_metrics == pytest.approx(recomputed_metrics, nan_ok=True)
    ledger = build_trade_ledger(stitched_results)
    boundary_rows = ledger.filter(pl.col("Date") == dates[10])
    assert boundary_rows.height == 1
    assert boundary_rows["Action"][0] == "Flip Long To Short"
    assert boundary_rows["Turnover"][0] == pytest.approx(2.0)


def test_walk_forward_optimise_threads_pairs_leg_weights_across_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify walk_forward_optimise threads pairs leg weights between folds."""
    pairs_data = _make_pairs_transition_data(24)
    original_run_backtest = run_backtest
    original_eval = evaluate_validation_period
    received_prior_positions: list[PortfolioPosition | None] = []

    def guided_run_backtest(
        frame: pl.DataFrame,
        strategy_type: str,
        params: dict[str, Any],
        tickers: str | list[str],
        **kwargs: Any,
    ) -> tuple[pl.DataFrame, dict[str, float]]:
        results, metrics = original_run_backtest(
            frame, strategy_type, params, tickers, **kwargs
        )
        # n_folds=2 on 24 rows -> train1=8 rows, train2=16 rows.
        if len(frame) == 8:
            metrics = dict(metrics)
            metrics["Sharpe Ratio"] = 2.0 if params["window"] == 5 else 1.0
        elif len(frame) == 16:
            metrics = dict(metrics)
            metrics["Sharpe Ratio"] = 2.0 if params["window"] == 8 else 1.0
        return results, metrics

    def spying_eval(*args: Any, **kwargs: Any) -> tuple[pl.DataFrame, dict[str, float]]:
        received_prior_positions.append(kwargs.get("prior_position"))
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        guided_run_backtest,
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.evaluate_validation_period",
        spying_eval,
    )

    _, _, fold_results = walk_forward_optimise(
        pairs_data,
        "Pairs Trading",
        {"window": [5, 8], "entry_z_score": [1.0], "exit_z_score": [0.2]},
        ["A", "B"],
        n_folds=2,
        transaction_cost_bps=15.0,
        slippage_bps=5.0,
    )

    fold_1_oos = fold_results[0]["oos_results"]
    fold_2_oos = fold_results[1]["oos_results"]
    expected_prior = PortfolioPosition(
        signal=float(fold_1_oos["signal"][-1]),
        leg_1_weight=float(fold_1_oos["leg_1_weight"][-1]),
        leg_2_weight=float(fold_1_oos["leg_2_weight"][-1]),
    )
    assert received_prior_positions == [None, expected_prior]
    expected_turnover = abs(
        float(fold_2_oos["leg_1_weight"][0]) - float(fold_1_oos["leg_1_weight"][-1])
    ) + abs(
        float(fold_2_oos["leg_2_weight"][0]) - float(fold_1_oos["leg_2_weight"][-1])
    )
    assert expected_turnover > 0.0
    assert fold_2_oos["trade_turnover"][0] == pytest.approx(expected_turnover)
    assert fold_2_oos["pair_turnover"][0] == pytest.approx(expected_turnover)


def test_fold_transition_reconciliation_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid prior positions or missing columns raise ValueError."""
    with pytest.raises(ValueError, match="signal must be finite"):
        PortfolioPosition(signal=float("nan"))

    with pytest.raises(ValueError, match="must both be set or both be None"):
        PortfolioPosition(signal=1.0, leg_1_weight=0.5, leg_2_weight=None)

    with pytest.raises(ValueError, match="leg_1_weight must be finite"):
        PortfolioPosition(signal=1.0, leg_1_weight=float("inf"), leg_2_weight=-0.5)

    pairs_data = _make_pairs_transition_data(20)
    with pytest.raises(ValueError, match="leg_1_weight and leg_2_weight"):
        evaluate_validation_period(
            pairs_data,
            pairs_data[14:],
            "Pairs Trading",
            {"window": 5, "entry_z_score": 1.0, "exit_z_score": 0.2},
            ["A", "B"],
            prior_position=PortfolioPosition(signal=1.0),
        )

    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(6)]
    single_data = pl.DataFrame({"Date": dates, "Close": [100.0 + i for i in range(6)]})
    incomplete_results = pl.DataFrame(
        {
            "Date": dates,
            "signal": [1.0] * 6,
            "position_change": [0.0] * 6,
            "gross_strategy_returns": [0.0] * 6,
            "trade_turnover": [0.0] * 6,
            "transaction_costs": [0.0] * 6,
            "strategy_returns": [0.0] * 6,
        }
    )
    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        lambda *_args, **_kwargs: (incomplete_results, {"Sharpe Ratio": 1.0}),
    )
    with pytest.raises(ValueError, match="asset_returns"):
        evaluate_validation_period(
            single_data,
            single_data[3:],
            "Buy and Hold",
            {},
            "TEST",
            prior_position=PortfolioPosition(signal=0.0),
        )

    with pytest.raises(ValueError, match="cannot accept leg_1_weight"):
        evaluate_validation_period(
            single_data,
            single_data[3:],
            "Buy and Hold",
            {},
            "TEST",
            prior_position=PortfolioPosition(
                signal=1.0, leg_1_weight=0.5, leg_2_weight=-0.5
            ),
        )


def test_fold_transitions_handle_int64_signals_unsorted_dates_and_nulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Int64 signals stitch cleanly and edge cases stay safe."""
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(9)]
    closes = [100.0 + i * 2.0 for i in range(9)]
    data = pl.DataFrame({"Date": dates, "Close": closes})

    def int64_run_backtest(
        frame: pl.DataFrame,
        strategy_type: str,
        params: dict[str, Any],
        tickers: str | list[str],
        **kwargs: Any,
    ) -> tuple[pl.DataFrame, dict[str, float]]:
        n_rows = len(frame)
        signal_val = 1 if params.get("short_window", 1) == 1 else -1
        signals = [signal_val] * n_rows
        pos_changes = [signal_val] + [0] * (n_rows - 1)
        asset_rets = [0.0] + [
            (frame["Close"][i] - frame["Close"][i - 1]) / frame["Close"][i - 1]
            for i in range(1, n_rows)
        ]
        gross = [0.0] + [signal_val * r for r in asset_rets[1:]]
        turnover = [abs(c) for c in pos_changes]
        cost_rate = (
            kwargs.get("transaction_cost_bps", 5.0) + kwargs.get("slippage_bps", 3.0)
        ) / 10_000
        tx_costs = [t * cost_rate for t in turnover]
        net = [g - c for g, c in zip(gross, tx_costs, strict=True)]
        results = pl.DataFrame(
            {
                "Date": frame["Date"],
                "Close": frame["Close"],
                "signal": pl.Series(signals, dtype=pl.Int64),
                "position_change": pl.Series(pos_changes, dtype=pl.Int64),
                "asset_returns": pl.Series(asset_rets, dtype=pl.Float64),
                "gross_strategy_returns": pl.Series(gross, dtype=pl.Float64),
                "trade_turnover": pl.Series(turnover, dtype=pl.Int64),
                "transaction_costs": pl.Series(tx_costs, dtype=pl.Float64),
                "strategy_returns": pl.Series(net, dtype=pl.Float64),
                "gross_cumulative_returns": pl.Series([1.0] * n_rows, dtype=pl.Float64),
                "cumulative_returns": pl.Series([1.0] * n_rows, dtype=pl.Float64),
                "equity_curve": pl.Series([100_000.0] * n_rows, dtype=pl.Float64),
                "cumulative_transaction_costs": pl.Series(
                    [0.0] * n_rows, dtype=pl.Float64
                ),
            }
        )
        return results, {"Sharpe Ratio": 1.0}

    monkeypatch.setattr(
        "quant_trading_strategy_backtester.optimiser.run_backtest",
        int64_run_backtest,
    )

    fold_1_res, _ = evaluate_validation_period(
        data[:6],
        data[3:6],
        "Moving Average Crossover",
        {"short_window": 1, "long_window": 2},
        "TEST",
    )
    # Pass unsorted validation_data to verify maintain_order keeps row 0 first.
    unsorted_fold_2 = data[6:9].reverse()
    fold_2_res, _ = evaluate_validation_period(
        data[:9],
        unsorted_fold_2,
        "Moving Average Crossover",
        {"short_window": 2, "long_window": 3},
        "TEST",
        prior_position=_extract_closing_position(fold_1_res),
    )

    assert fold_1_res.schema == fold_2_res.schema
    assert fold_2_res["Date"][0] == dates[6]
    assert fold_2_res["position_change"][0] == pytest.approx(-2.0)
    assert fold_2_res["trade_turnover"][0] == pytest.approx(2.0)

    # Both direct vertical concat and stitch_walk_forward_oos_results succeed.
    assert len(pl.concat([fold_1_res, fold_2_res], how="vertical")) == 6
    stitched, _ = stitch_walk_forward_oos_results(
        [
            {"fold": 1, "oos_results": fold_1_res},
            {"fold": 2, "oos_results": fold_2_res},
        ]
    )
    assert len(stitched) == 6

    # Direct stitch with mixed Int64/Float64 fold frames also succeeds.
    int64_fold = fold_1_res.with_columns(
        pl.col("position_change").cast(pl.Int64),
        pl.col("trade_turnover").cast(pl.Int64),
    )
    mixed_stitched, _ = stitch_walk_forward_oos_results(
        [
            {"fold": 1, "oos_results": int64_fold},
            {"fold": 2, "oos_results": fold_2_res},
        ]
    )
    assert len(mixed_stitched) == 6

    # None / NaN closing signals and weights fall back cleanly to 0.0.
    null_closing = pl.DataFrame(
        {
            "signal": pl.Series([1.0, None], dtype=pl.Float64),
            "leg_1_weight": pl.Series([0.5, float("nan")], dtype=pl.Float64),
            "leg_2_weight": pl.Series([-0.5, None], dtype=pl.Float64),
        }
    )
    assert _extract_closing_position(null_closing) == PortfolioPosition(
        signal=0.0,
        leg_1_weight=0.0,
        leg_2_weight=0.0,
    )
