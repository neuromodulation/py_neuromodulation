"""
Reference:  B Hjorth
            EEG analysis based on time domain properties
            Electroencephalogr Clin Neurophysiol. 1970 Sep;29(3):306-10.
            DOI: 10.1016/0013-4694(70)90143-4
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

    def calc_feature(self, data: np.ndarray) -> dict:
        var = np.var(data, axis=-1)
        deriv1 = np.diff(data, axis=-1)
        deriv2 = np.diff(deriv1, axis=-1)
        deriv1_var = np.var(deriv1, axis=-1)
        deriv2_var = np.var(deriv2, axis=-1)
        deriv1_mobility = np.sqrt(deriv2_var / deriv1_var)

        activity = np.nan_to_num(var)
        mobility = np.nan_to_num(np.sqrt(deriv1_var / var))
        complexity = np.nan_to_num(deriv1_mobility / mobility)

        feature_results = {}
        for ch_idx, ch_name in enumerate(self.ch_names):
            feature_results[f"{ch_name}_RawHjorth_Activity"] = activity[ch_idx]
            feature_results[f"{ch_name}_RawHjorth_Mobility"] = mobility[ch_idx]
            feature_results[f"{ch_name}_RawHjorth_Complexity"] = complexity[ch_idx]

        return feature_results


class Raw(NMFeature):
    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.ch_names = ch_names

    def calc_feature(self, data: np.ndarray) -> dict:
        feature_results = {}

        for ch_idx, ch_name in enumerate(self.ch_names):
            feature_results["_".join([ch_name, "raw"])] = data[ch_idx, -1]

        return feature_results
