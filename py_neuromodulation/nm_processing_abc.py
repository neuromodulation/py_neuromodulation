"""Module that contains the preprocessor abstract base class."""
from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        ...
