from scipy import signal
import numpy as np
from typing import Iterable

from py_neuromodulation import nm_features_abc


class CoherenceObject:
    def __init__(
        self,
        sfreq,
        window,
        fbands,
        fband_names,
        ch_1_name,
        ch_2_name,
        ch_1_idx,
        ch_2_idx,
        coh,
        icoh,
    ) -> None:
        self.sfreq = sfreq
        self.window = window
        self.Pxx = None
        self.Pyy = None
        self.Pxy = None
        self.f = None
        self.coh = coh
        self.icoh = icoh
        self.coh_val = None
        self.icoh_val = None
        self.ch_1 = ch_1_name
        self.ch_2 = ch_2_name
        self.ch_1_idx = ch_1_idx
        self.ch_2_idx = ch_2_idx
        self.fbands = fbands  # list of lists, e.g. [[10, 15], [15, 20]]
        self.fband_names = fband_names

    def get_coh(self, features_compute, x, y):
        self.f, self.Pxx = signal.welch(x, self.sfreq, self.window, nperseg=128)
        self.Pyy = signal.welch(y, self.sfreq, self.window, nperseg=128)[1]
        self.Pxy = signal.csd(x, y, self.sfreq, self.window, nperseg=128)[1]

        if self.coh is True:
            self.coh_val = np.abs(self.Pxy**2) / (self.Pxx * self.Pyy)
        if self.icoh is True:
            self.icoh_val = np.array(self.Pxy / (self.Pxx * self.Pyy)).imag

        for idx, fband in enumerate(self.fbands):
            if self.coh is True:
                feature_calc = np.mean(
                    self.coh_val[
                        np.bitwise_and(self.f > fband[0], self.f < fband[1])
                    ]
                )
                feature_name = "_".join(
                    ["coh", self.ch_1, "to", self.ch_2, self.fband_names[idx]]
                )
                features_compute[feature_name] = feature_calc
            if self.icoh is True:
                feature_calc = np.mean(
                    self.icoh_val[
                        np.bitwise_and(self.f > fband[0], self.f < fband[1])
                    ]
                )
                feature_name = "_".join(
                    ["icoh", self.ch_1, "to", self.ch_2, self.fband_names[idx]]
                )
                features_compute[feature_name] = feature_calc
        return features_compute


class NM_Coherence(nm_features_abc.Feature):

    coherence_objects: Iterable[CoherenceObject] = []

    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names

        for idx_coh in range(len(self.s["coherence"]["channels"])):
            fband_names = self.s["coherence"]["frequency_bands"][idx_coh]
            fband_specs = []
            for band_name in fband_names:
                fband_specs.append(self.s["frequency_ranges_hz"][band_name])

            ch_1_name = self.s["coherence"]["channels"][idx_coh][0]
            ch_1_name_reref = [
                ch for ch in self.ch_names if ch.startswith(ch_1_name)
            ][0]
            ch_1_idx = self.ch_names.index(ch_1_name_reref)

            ch_2_name = self.s["coherence"]["channels"][idx_coh][1]
            ch_2_name_reref = [
                ch for ch in self.ch_names if ch.startswith(ch_2_name)
            ][0]
            ch_2_idx = self.ch_names.index(ch_2_name_reref)

            self.coherence_objects.append(
                CoherenceObject(
                    sfreq,
                    self.s["coherence"]["params"][idx_coh]["window"],
                    fband_specs,
                    fband_names,
                    ch_1_name,
                    ch_2_name,
                    ch_1_idx,
                    ch_2_idx,
                    self.s["coherence"]["method"][idx_coh]["coh"],
                    self.s["coherence"]["method"][idx_coh]["icoh"],
                )
            )

    def calc_feature(
        self, data: np.array, features_compute: dict, ch_names: Iterable[str]
    ) -> dict:
        for coh_obj in self.coherence_objects:
            features_compute = coh_obj.get_coh(
                features_compute,
                data[coh_obj.ch_1_idx, :],
                data[coh_obj.ch_2_idx, :],
            )

        return features_compute
