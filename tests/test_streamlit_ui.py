"""Test Streamlit input behaviour."""

from quant_trading_strategy_backtester import streamlit_ui


def test_walk_forward_is_disabled_for_automatic_ticker_selection(monkeypatch) -> None:
    """Verify automatic selection cannot enable leaked fold reporting."""

    class Sidebar:
        def checkbox(self, label: str, **kwargs) -> bool:
            if label == "Optimise Strategy Parameters":
                return True

            assert label == "Use Walk-Forward Validation"
            assert kwargs["disabled"] is True
            return False

    monkeypatch.setattr(streamlit_ui.st, "sidebar", Sidebar())

    optimise, walk_forward, _ = streamlit_ui.get_user_inputs_for_strategy_params(
        "Mean Reversion",
        auto_select_tickers=True,
    )

    assert optimise is True
    assert walk_forward is False
