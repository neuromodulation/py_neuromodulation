import numpy as np
from typing import Iterable
import nolds

from py_neuromodulation import nm_features_abc


class Nolds(nm_features_abc.Feature):

    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.s = settings
        self.ch_names = ch_names

    def calc_feature(
        self, data: np.array, features_compute: dict,
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.s["nolds_features"]["sample_entropy"]:
                features_compute[f"{ch_name}_sample_entropy"] = nolds.sampen(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["correlation_dimension"]:
                features_compute[
                    f"{ch_name}_correlation_dimension"
                ] = nolds.corr_dim(data[ch_idx, :], emb_dim=2)
            if self.s["nolds_features"]["lyapunov_exponent"]:
                features_compute[f"{ch_name}_lyapunov_exponent"] = nolds.lyap_r(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["hurst_exponent"]:
                features_compute[f"{ch_name}_hurst_exponent"] = nolds.hurst_rs(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["detrended_fluctutaion_analysis"]:
                features_compute[
                    f"{ch_name}_detrended_fluctutaion_analysis"
                ] = nolds.dfa(data[ch_idx, :])

        return features_compute
