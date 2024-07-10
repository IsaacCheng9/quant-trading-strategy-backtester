import pytest
import pandas as pd


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
