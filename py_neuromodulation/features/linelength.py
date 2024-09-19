import numpy as np
from collections.abc import Sequence

from py_neuromodulation.features.feature_processor import NMFeature


class LineLength(NMFeature):
    def __init__(self, settings: dict, ch_names: Sequence[str], sfreq: float) -> None:
        self.ch_names = ch_names

    def calc_feature(self, data: np.ndarray) -> dict:

        line_length = np.mean(
            np.abs(np.diff(data, axis=-1)) / (data.shape[1] - 1), axis=-1
        )

        feature_results = {}
        for ch_idx, ch_name in enumerate(self.ch_names):
            feature_results[f"{ch_name}_LineLength"] = line_length[ch_idx]

        return feature_results
