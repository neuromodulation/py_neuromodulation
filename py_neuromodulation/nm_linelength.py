import numpy as np
from typing import Iterable

from py_neuromodulation import nm_features_abc


class LineLength(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.ch_names = ch_names

    @staticmethod
    def get_line_length(x: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(np.diff(x)) / (x.shape[0] - 1))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute[
                "_".join([ch_name, "LineLength"])
            ] = self.get_line_length(data[ch_idx, :])

        return features_compute
