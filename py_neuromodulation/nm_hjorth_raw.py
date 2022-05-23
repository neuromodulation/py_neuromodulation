import enum
import numpy as np
from typing import Iterable

from py_neuromodulation import nm_features_abc


class Hjorth(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.ch_names = ch_names

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute[
                "_".join([ch_name, "RawHjorth_Activity"])
            ] = np.var(data[ch_idx, :])
            deriv_variance = np.var(np.diff(data[ch_idx, :]))
            mobility = np.sqrt(deriv_variance / np.var(data[ch_idx, :]))
            features_compute[
                "_".join([ch_name, "RawHjorth_Mobility"])
            ] = mobility

            dat_deriv_2_var = np.var(np.diff(np.diff(data[ch_idx, :])))
            deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)
            features_compute["_".join([ch_name, "RawHjorth_Complexity"])] = (
                deriv_mobility / mobility
            )

        return features_compute


class Raw(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.ch_names = ch_names

    def calc_feature(
        self,
        data: np.array,
        features_compute: dict,
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute["_".join([ch_name, "raw"])] = data[ch_idx, -1]

        return features_compute
