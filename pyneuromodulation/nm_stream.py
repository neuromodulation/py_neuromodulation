from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class PNStream(ABC):

    @abstractmethod
    def run(self, ieeg_batch: np.array) -> pd.Series:
        pass

    @abstractmethod
    def set_rereference(self) -> None:
        pass

    @abstractmethod
    def set_resampling(self) -> None:
        pass

    @abstractmethod
    def set_features(self) -> None:
        pass

    @abstractmethod
    def set_run(self) -> None:
        pass

    @abstractmethod
    def set_fs(self, fs: int) -> None:
        pass

    @abstractmethod 
    def set_linenoise(self, line_noise: int) -> None:
        pass

    @abstractmethod
    def set_settings(self, PATH_SETTINGS: str) -> None:
        pass

    @abstractmethod
    def set_channels(self, PATH_CHANNELS: str) -> None:
        pass
