"""
Validate trading strategy parameter dictionaries.
"""

from collections.abc import Mapping
from typing import Any


def validate_strategy_params(strategy_type: str, params: Mapping[str, Any]) -> None:
    """
    Validate strategy parameters for a concrete strategy instance.

    Args:
        strategy_type: The strategy name.
        params: Scalar parameter values for the strategy.

    Raises:
        ValueError: If any required parameter is missing or invalid.
    """
    match strategy_type:
        case "Buy and Hold":
            return
        case "Moving Average Crossover":
            short_window = _require_positive_int(params, "short_window")
            long_window = _require_positive_int(params, "long_window")
            if short_window >= long_window:
                raise ValueError("short_window must be less than long_window")
        case "Mean Reversion":
            _require_positive_int(params, "window")
            _require_positive_float(params, "std_dev")
        case "Pairs Trading":
            _require_positive_int(params, "window")
            entry_z_score = _require_positive_float(params, "entry_z_score")
            exit_z_score = _require_positive_float(params, "exit_z_score")
            if exit_z_score >= entry_z_score:
                raise ValueError("exit_z_score must be less than entry_z_score")
        case _:
            raise ValueError(f"Unexpected strategy type: {strategy_type}")


def is_valid_strategy_params(strategy_type: str, params: Mapping[str, Any]) -> bool:
    """
    Return whether a strategy parameter dictionary is valid.

    Args:
        strategy_type: The strategy name.
        params: Scalar parameter values for the strategy.

    Returns:
        True when the parameter dictionary passes validation.
    """
    try:
        validate_strategy_params(strategy_type, params)
    except ValueError:
        return False

    return True


def _require_positive_int(params: Mapping[str, Any], name: str) -> int:
    """Return a required positive integer parameter."""
    value = _require_number(params, name)
    int_value = int(value)
    if int_value != value:
        raise ValueError(f"{name} must be an integer")
    if int_value <= 0:
        raise ValueError(f"{name} must be positive")

    return int_value


def _require_positive_float(params: Mapping[str, Any], name: str) -> float:
    """Return a required positive numeric parameter."""
    value = float(_require_number(params, name))
    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _require_number(params: Mapping[str, Any], name: str) -> int | float:
    """Return a required non-boolean numeric parameter."""
    if name not in params:
        raise ValueError(f"Missing required parameter: {name}")

    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")

    return value
