import numpy as np
from typing import Iterable

from py_neuromodulation.nm_features import NMFeature


class LineLength(NMFeature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.ch_names = ch_names

    @staticmethod
    def get_line_length(x: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(np.diff(x)) / (x.shape[0] - 1))

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        # no settings to be checked
        pass

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute[
                "_".join([ch_name, "LineLength"])
            ] = self.get_line_length(data[ch_idx, :])

        return features_compute
