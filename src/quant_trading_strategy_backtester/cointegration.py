"""
Evaluate pair cointegration for pairs-trading selection and diagnostics.
"""

from dataclasses import dataclass
import math

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import coint

COINTEGRATION_P_VALUE_THRESHOLD = 0.05
MIN_COINTEGRATION_OBSERVATIONS = 30


@dataclass(frozen=True)
class CointegrationResult:
    """Store the outcome of an Engle-Granger cointegration test."""

    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_value_1pct: float
    critical_value_5pct: float
    critical_value_10pct: float
    reason: str | None = None


def evaluate_cointegration(
    data: pl.DataFrame,
    p_value_threshold: float = COINTEGRATION_P_VALUE_THRESHOLD,
) -> CointegrationResult:
    """
    Run an Engle-Granger cointegration test on a candidate pair.

    Args:
        data: Pair price data containing Close_1 and Close_2.
        p_value_threshold: Maximum p-value accepted as cointegrated.

    Returns:
        A cointegration test result with p-value and critical values.

    Raises:
        ValueError: If required price columns are missing.
    """
    if "Close_1" not in data.columns or "Close_2" not in data.columns:
        raise ValueError("Cointegration test requires Close_1 and Close_2 columns")

    cleaned_data = data.select(["Close_1", "Close_2"]).drop_nulls()
    if cleaned_data.height < MIN_COINTEGRATION_OBSERVATIONS:
        return _failed_result("Not enough observations for cointegration test")

    close_1 = cleaned_data["Close_1"].cast(pl.Float64).to_numpy()
    close_2 = cleaned_data["Close_2"].cast(pl.Float64).to_numpy()
    if not bool(np.isfinite(close_1).all() and np.isfinite(close_2).all()):
        return _failed_result("Price series contains non-finite values")
    if np.isclose(float(np.std(close_1)), 0.0) or np.isclose(
        float(np.std(close_2)), 0.0
    ):
        return _failed_result("Price series is constant")

    try:
        test_statistic, p_value, critical_values = coint(
            close_1,
            close_2,
            trend="c",
            autolag="aic",
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _failed_result(f"Cointegration test failed: {exc}")

    p_value = float(p_value)
    if not math.isfinite(p_value):
        return _failed_result("Cointegration test returned a non-finite p-value")

    return CointegrationResult(
        is_cointegrated=p_value <= p_value_threshold,
        p_value=p_value,
        test_statistic=float(test_statistic),
        critical_value_1pct=float(critical_values[0]),
        critical_value_5pct=float(critical_values[1]),
        critical_value_10pct=float(critical_values[2]),
    )


def _failed_result(reason: str) -> CointegrationResult:
    """Return a failed cointegration result with undefined statistics."""
    return CointegrationResult(
        is_cointegrated=False,
        p_value=float("nan"),
        test_statistic=float("nan"),
        critical_value_1pct=float("nan"),
        critical_value_5pct=float("nan"),
        critical_value_10pct=float("nan"),
        reason=reason,
    )
