from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data):
        raise NotImplementedError("Method 'generate_signals' must be implemented.")


class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0.0
        signals["short_mavg"] = (
            data["Close"]
            .rolling(window=self.short_window, min_periods=1, center=False)
            .mean()
        )
        signals["long_mavg"] = (
            data["Close"]
            .rolling(window=self.long_window, min_periods=1, center=False)
            .mean()
        )
        signals.loc[signals["short_mavg"] > signals["long_mavg"], "signal"] = 1.0
        signals["positions"] = signals["signal"].diff()

        return signals
