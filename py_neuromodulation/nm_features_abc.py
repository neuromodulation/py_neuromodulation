from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Iterable


class Feature(ABC):
    @abstractmethod
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: int | float
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        """Method to check passed settings"""
        pass

    @abstractmethod
    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        """
        Feature calculation method. Each method needs to loop through all channels

        Parameters
        ----------
        data : np.ndarray
            (channels, time)
        features_compute : dict
        ch_names : Iterable[str]

        Returns
        -------
        dict
        """
        pass
