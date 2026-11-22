"""Tests for visualisation helpers."""

import datetime
import math

import pandas as pd
import polars as pl
import pytest

from quant_trading_strategy_backtester import visualisation
from quant_trading_strategy_backtester.visualisation import (
    calculate_monthly_returns,
    display_returns_by_month,
)


def test_calculate_monthly_returns_includes_day_one_return() -> None:
    """Verify day-1 return is included in single-month and rolling returns."""
    daily_returns = [-0.01, 0.05, 0.02]
    results = pl.DataFrame(
        {
            "Date": [
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 4),
            ],
            "strategy_returns": daily_returns,
        }
    )

    monthly = calculate_monthly_returns(results)

    expected = round((math.prod(1 + r for r in daily_returns) - 1) * 100, 2)
    assert monthly["Month (YYYY-MM)"].to_list() == ["2024-01"]
    assert monthly["Monthly Return (%)"].to_list() == pytest.approx([expected])
    assert monthly["Rolling Return (%)"].to_list() == pytest.approx([expected])
    assert expected == 6.03


def test_calculate_monthly_returns_multi_month_captures_boundary_and_compounds() -> (
    None
):
    """Verify month-boundary returns and unrounded rolling compounding."""
    jan_returns = [-0.005, 0.02, 0.01]
    feb_returns = [0.04, -0.01]
    mar_returns = [0.0155, 0.01]
    all_returns = jan_returns + feb_returns + mar_returns

    results = pl.DataFrame(
        {
            "Date": [
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 15),
                datetime.date(2024, 1, 31),
                datetime.date(2024, 2, 1),
                datetime.date(2024, 2, 20),
                datetime.date(2024, 3, 1),
                datetime.date(2024, 3, 28),
            ],
            "strategy_returns": all_returns,
        }
    )

    monthly = calculate_monthly_returns(results)

    jan_factor = math.prod(1 + r for r in jan_returns)
    feb_factor = math.prod(1 + r for r in feb_returns)
    mar_factor = math.prod(1 + r for r in mar_returns)
    overall_return = round((math.prod(1 + r for r in all_returns) - 1) * 100, 2)

    assert monthly["Month (YYYY-MM)"].to_list() == ["2024-01", "2024-02", "2024-03"]
    assert monthly["Monthly Return (%)"].to_list() == pytest.approx(
        [
            round((jan_factor - 1) * 100, 2),
            round((feb_factor - 1) * 100, 2),
            round((mar_factor - 1) * 100, 2),
        ]
    )
    # Compounding unrounded factors gives 5.54% in 2024-02 rather than 5.53%.
    assert monthly["Rolling Return (%)"].to_list() == pytest.approx(
        [
            round((jan_factor - 1) * 100, 2),
            round((jan_factor * feb_factor - 1) * 100, 2),
            overall_return,
        ]
    )
    assert monthly["Rolling Return (%)"][1] == 5.54
    assert monthly["Rolling Return (%)"][-1] == overall_return


def test_calculate_monthly_returns_sorts_unsorted_dates() -> None:
    """Verify unsorted input dates produce chronologically ordered months."""
    results = pl.DataFrame(
        {
            "Date": [
                datetime.date(2024, 3, 15),
                datetime.date(2024, 1, 20),
                datetime.date(2024, 2, 2),
                datetime.date(2024, 1, 2),
                datetime.date(2024, 3, 1),
                datetime.date(2024, 2, 1),
            ],
            "strategy_returns": [0.01, 0.02, -0.01, -0.005, 0.0155, 0.04],
        }
    )

    monthly = calculate_monthly_returns(results)

    jan_factor = (1 - 0.005) * (1 + 0.02)
    feb_factor = (1 + 0.04) * (1 - 0.01)
    mar_factor = (1 + 0.0155) * (1 + 0.01)

    assert monthly["Month (YYYY-MM)"].to_list() == ["2024-01", "2024-02", "2024-03"]
    assert monthly["Monthly Return (%)"].to_list() == pytest.approx(
        [
            round((jan_factor - 1) * 100, 2),
            round((feb_factor - 1) * 100, 2),
            round((mar_factor - 1) * 100, 2),
        ]
    )
    assert monthly["Rolling Return (%)"].to_list() == pytest.approx(
        [
            round((jan_factor - 1) * 100, 2),
            round((jan_factor * feb_factor - 1) * 100, 2),
            round((jan_factor * feb_factor * mar_factor - 1) * 100, 2),
        ]
    )


def test_calculate_monthly_returns_empty_dataframe() -> None:
    """Verify empty input returns an empty DataFrame with expected columns."""
    empty_results = pl.DataFrame(
        schema={"Date": pl.Date, "strategy_returns": pl.Float64}
    )

    monthly = calculate_monthly_returns(empty_results)

    assert monthly.is_empty()
    assert monthly.columns == [
        "Month (YYYY-MM)",
        "Monthly Return (%)",
        "Rolling Return (%)",
    ]


def test_display_returns_by_month_renders_calculated_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify display_returns_by_month delegates to calculate_monthly_returns."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        visualisation.st,
        "subheader",
        lambda text: captured.setdefault("subheader", text),
    )
    monkeypatch.setattr(
        visualisation.st,
        "write",
        lambda text: captured.setdefault("write", text),
    )
    monkeypatch.setattr(
        visualisation.st,
        "dataframe",
        lambda df, **kwargs: captured.update({"df": df, "kwargs": kwargs}),
    )

    results = pl.DataFrame(
        {
            "Date": [datetime.date(2024, 1, 2), datetime.date(2024, 2, 1)],
            "strategy_returns": [0.02, 0.03],
        }
    )
    display_returns_by_month(results)

    assert captured["subheader"] == "Returns by Month"
    assert isinstance(captured["df"], pd.DataFrame)
    assert captured["df"].to_dict(orient="list") == {
        "Month (YYYY-MM)": ["2024-01", "2024-02"],
        "Monthly Return (%)": [2.0, 3.0],
        "Rolling Return (%)": [2.0, 5.06],
    }
    assert captured["kwargs"] == {"width": "content", "hide_index": True}

    captured.clear()
    display_returns_by_month(
        pl.DataFrame(schema={"Date": pl.Date, "strategy_returns": pl.Float64})
    )
    assert captured["write"] == "No data available for monthly performance calculation."


def test_calculate_monthly_returns_clamps_bankrupt_compounding() -> None:
    """Verify bankruptcy clamps monthly and rolling returns without flips."""
    results = pl.DataFrame(
        {
            "Date": [
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 15),
                datetime.date(2024, 1, 20),
                datetime.date(2024, 2, 1),
                datetime.date(2024, 3, 1),
            ],
            "strategy_returns": [0.05, -1.5, 0.2, 0.1, -1.5],
        }
    )

    monthly = calculate_monthly_returns(results)

    assert monthly["Month (YYYY-MM)"].to_list() == ["2024-01", "2024-02", "2024-03"]
    assert monthly["Monthly Return (%)"].to_list() == pytest.approx([-100.0, 0.0, 0.0])
    assert monthly["Rolling Return (%)"].to_list() == pytest.approx(
        [-100.0, -100.0, -100.0]
    )
