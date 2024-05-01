import numpy as np
from collections.abc import Iterable

from py_neuromodulation.nm_features_abc import Feature


class Hjorth(Feature):
    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.settings = settings
        self.ch_names = ch_names

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ):
        # no settings to test
        pass

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            features_compute["_".join([ch_name, "RawHjorth_Activity"])] = np.nan_to_num(
                np.var(data[ch_idx, :])
            )
            deriv_variance = np.nan_to_num(np.var(np.diff(data[ch_idx, :])))
            mobility = np.nan_to_num(np.sqrt(deriv_variance / np.var(data[ch_idx, :])))
            features_compute["_".join([ch_name, "RawHjorth_Mobility"])] = mobility

            dat_deriv_2_var = np.nan_to_num(np.var(np.diff(np.diff(data[ch_idx, :]))))
            deriv_mobility = np.nan_to_num(np.sqrt(dat_deriv_2_var / deriv_variance))
            features_compute["_".join([ch_name, "RawHjorth_Complexity"])] = (
                np.nan_to_num(deriv_mobility / mobility)
            )

        return features_compute


class Raw(Feature):
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

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ):
        # no settings to test
        pass
