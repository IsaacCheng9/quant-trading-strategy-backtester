"""Test Streamlit sidebar input behaviour."""

import datetime
from typing import Any

from streamlit.testing.v1 import AppTest

from quant_trading_strategy_backtester import streamlit_ui
from quant_trading_strategy_backtester.utils import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSACTION_COST_BPS,
)


class _FakeStreamlitHarness:
    """Capture widget calls inside and outside st.sidebar.form."""

    def __init__(
        self,
        *,
        strategy_type: str = "Moving Average Crossover",
        auto_select_tickers: bool = False,
        optimise: bool = False,
        walk_forward: bool = False,
        submitted: bool = True,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    ) -> None:
        self.strategy_type = strategy_type
        self.auto_select_tickers = auto_select_tickers
        self.optimise = optimise
        self.walk_forward = walk_forward
        self.submitted = submitted
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.in_form = False
        self.form_keys: list[str] = []
        self.outside_widgets: list[str] = []
        self.inside_widgets: list[str] = []
        self.checkbox_kwargs: dict[str, dict[str, Any]] = {}
        self.number_input_defaults: dict[str, float] = {}
        self.sidebar = self

    def form(self, key: str) -> "_FakeStreamlitHarness":
        self.form_keys.append(key)
        return self

    def __enter__(self) -> None:
        self.in_form = True

    def __exit__(self, *_args: object) -> None:
        self.in_form = False

    def _record(self, label: str) -> None:
        if self.in_form:
            self.inside_widgets.append(label)
        else:
            self.outside_widgets.append(label)

    def selectbox(self, label: str, options: list[str], index: int = 0) -> str:
        self._record(label)
        assert self.strategy_type in options
        return self.strategy_type

    def checkbox(self, label: str, **kwargs: Any) -> bool:
        self._record(label)
        self.checkbox_kwargs[label] = kwargs
        if label == "Optimise Strategy Parameters":
            return self.optimise
        if label == "Use Walk-Forward Validation":
            return self.walk_forward
        return self.auto_select_tickers

    def text_input(self, label: str, value: str = "") -> str:
        self._record(label)
        return value

    def date_input(self, label: str, value: datetime.date) -> datetime.date:
        self._record(label)
        return value

    def slider(self, label: str, **kwargs: Any) -> int | float:
        self._record(label)
        return kwargs["value"]

    def subheader(self, label: str) -> None:
        self._record(label)

    def number_input(self, label: str, **kwargs: Any) -> float:
        self._record(label)
        self.number_input_defaults[label] = float(kwargs["value"])
        if label == "Initial Capital ($)":
            return self.initial_capital
        if label == "Transaction Cost (bps)":
            return self.transaction_cost_bps
        if label == "Slippage (bps)":
            return self.slippage_bps
        raise AssertionError(f"Unexpected number_input: {label}")

    def form_submit_button(self, label: str) -> bool:
        self._record(label)
        return self.submitted


def test_get_sidebar_inputs_separates_dynamic_controls_and_form_fields(
    monkeypatch,
) -> None:
    """Verify layout controls stay outside st.form and inputs stay inside."""
    harness = _FakeStreamlitHarness(
        strategy_type="Moving Average Crossover",
        auto_select_tickers=False,
        optimise=False,
        submitted=True,
        initial_capital=250_000.0,
        transaction_cost_bps=12.5,
        slippage_bps=4.0,
    )
    monkeypatch.setattr(streamlit_ui, "st", harness)

    config, submitted = streamlit_ui.get_sidebar_inputs()

    assert submitted is True
    assert harness.form_keys == ["backtest_config"]
    assert harness.outside_widgets == [
        "Strategy Type",
        "Explore Current Top 100 S&P 500 Companies",
        "Optimise Strategy Parameters",
    ]
    assert harness.inside_widgets == [
        "Ticker Symbol",
        "Start Date",
        "End Date",
        "Short Window (Days)",
        "Long Window (Days)",
        "Trading Assumptions",
        "Initial Capital ($)",
        "Transaction Cost (bps)",
        "Slippage (bps)",
        "Run Backtest",
    ]
    assert harness.number_input_defaults == {
        "Initial Capital ($)": DEFAULT_INITIAL_CAPITAL,
        "Transaction Cost (bps)": DEFAULT_TRANSACTION_COST_BPS,
        "Slippage (bps)": DEFAULT_SLIPPAGE_BPS,
    }
    assert config.ticker == "AAPL"
    assert config.strategy_params == {"short_window": 20, "long_window": 50}
    assert config.initial_capital == 250_000.0
    assert config.transaction_cost_bps == 12.5
    assert config.slippage_bps == 4.0


def test_get_sidebar_inputs_forces_walk_forward_false_when_auto_selecting(
    monkeypatch,
) -> None:
    """Verify walk-forward is disabled and forced False with auto-selection."""
    harness = _FakeStreamlitHarness(
        strategy_type="Pairs Trading",
        auto_select_tickers=True,
        optimise=True,
        walk_forward=True,
        submitted=False,
    )
    monkeypatch.setattr(streamlit_ui, "st", harness)

    config, submitted = streamlit_ui.get_sidebar_inputs()

    assert submitted is False
    assert config.ticker is None
    assert config.auto_select_tickers is True
    assert config.optimise is True
    assert config.walk_forward is False
    assert harness.checkbox_kwargs["Use Walk-Forward Validation"]["disabled"] is True
    assert "Ticker Symbol 1" not in harness.inside_widgets
    assert "Ticker Symbol 2" not in harness.inside_widgets
    assert "Use Walk-Forward Validation" in harness.inside_widgets
    assert "Run Backtest" in harness.inside_widgets


def _render_sidebar_for_apptest() -> None:
    """Render sidebar inputs and validate fixed params inside AppTest."""
    # AppTest.from_function executes the function body in an isolated module.
    import streamlit as st

    from quant_trading_strategy_backtester.strategy_params import (
        validate_strategy_params,
    )
    from quant_trading_strategy_backtester.streamlit_ui import get_sidebar_inputs

    config, submitted = get_sidebar_inputs()
    st.session_state["last_config"] = config
    st.session_state["last_submitted"] = submitted
    if not config.optimise:
        try:
            validate_strategy_params(config.strategy_type, config.strategy_params)
        except ValueError as exc:
            st.error(str(exc))


def test_apptest_mac_form_preserves_long_window_and_surfaces_invalid_params() -> None:
    """Verify MAC sliders preserve submitted values and surface errors."""
    app_test = AppTest.from_function(_render_sidebar_for_apptest).run()
    app_test.sidebar.selectbox[0].select("Moving Average Crossover").run()

    # Valid multi-slider update must not reset Long Window (Days) to default.
    app_test.sidebar.slider[0].set_value(10)
    app_test.sidebar.slider[1].set_value(120)
    app_test.sidebar.button[0].click().run()
    assert app_test.session_state["last_submitted"] is True
    assert app_test.session_state["last_config"].strategy_params == {
        "short_window": 10,
        "long_window": 120,
    }
    assert len(app_test.error) == 0

    # Invalid submission must keep user values and surface validation error.
    app_test.sidebar.slider[0].set_value(40)
    app_test.sidebar.slider[1].set_value(25)
    app_test.sidebar.button[0].click().run()
    assert app_test.session_state["last_config"].strategy_params == {
        "short_window": 40,
        "long_window": 25,
    }
    assert len(app_test.error) == 1
    assert "short_window" in app_test.error[0].value


def test_apptest_pairs_form_preserves_exit_z_and_surfaces_invalid_params() -> None:
    """Verify Pairs sliders preserve submitted values and surface errors."""
    app_test = AppTest.from_function(_render_sidebar_for_apptest).run()
    app_test.sidebar.selectbox[0].select("Pairs Trading").run()

    # Valid multi-slider update must keep Exit Z-Score rather than resetting.
    app_test.sidebar.slider[1].set_value(1.2)
    app_test.sidebar.slider[2].set_value(0.8)
    app_test.sidebar.button[0].click().run()
    assert app_test.session_state["last_submitted"] is True
    assert app_test.session_state["last_config"].strategy_params == {
        "window": 50,
        "entry_z_score": 1.2,
        "exit_z_score": 0.8,
    }
    assert len(app_test.error) == 0

    # Invalid submission must keep user values and surface validation error.
    app_test.sidebar.slider[1].set_value(1.0)
    app_test.sidebar.slider[2].set_value(1.5)
    app_test.sidebar.button[0].click().run()
    assert app_test.session_state["last_config"].strategy_params == {
        "window": 50,
        "entry_z_score": 1.0,
        "exit_z_score": 1.5,
    }
    assert len(app_test.error) == 1
    assert "exit_z_score" in app_test.error[0].value
