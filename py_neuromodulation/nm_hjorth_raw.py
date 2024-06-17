"""
Reference: Oh, S. H., Lee, Y. R., & Kim, H. N. (2014).
            A novel EEG feature extraction method using Hjorth parameter.
            International Journal of Electronics and Electrical Engineering, 2(2), 106-110.
http://cspl.ee.pusan.ac.kr/sites/cspl/download/internation_c/ICAEE2014_02_OSH.pdf

"""

import numpy as np
from collections.abc import Iterable

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_settings import NMSettings


class Hjorth(NMFeature):
    def __init__(
        self, settings: NMSettings, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.ch_names = ch_names

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        var = np.var(data, axis=-1)
        deriv1 = np.diff(data, axis=-1)
        deriv2 = np.diff(deriv1, axis=-1)
        deriv1_var = np.var(deriv1, axis=-1)
        deriv2_var = np.var(deriv2, axis=-1)
        deriv1_mobility = np.sqrt(deriv2_var / deriv1_var)

        activity = np.nan_to_num(var)
        mobility = np.nan_to_num(np.sqrt(deriv1_var / var))
        complexity = np.nan_to_num(deriv1_mobility / mobility)

        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute[f"{ch_name}_RawHjorth_Activity"] = activity[ch_idx]
            features_compute[f"{ch_name}_RawHjorth_Mobility"] = mobility[ch_idx]
            features_compute[f"{ch_name}_RawHjorth_Complexity"] = complexity[ch_idx]

        return features_compute


class Raw(NMFeature):
    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.ch_names = ch_names

    def calc_feature(
        self,
        data: np.ndarray,
        features_compute: dict,
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute["_".join([ch_name, "raw"])] = data[ch_idx, -1]

        return features_compute
