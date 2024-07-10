from abc import ABC, abstractmethod

# import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data):
        raise NotImplementedError("Method 'generate_signals' must be implemented.")
