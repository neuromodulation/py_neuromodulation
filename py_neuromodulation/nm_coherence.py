from scipy import signal
import numpy as np
from typing import Iterable
import warnings

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
        features_coh,
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
        self.features_coh = features_coh

    def get_coh(self, features_compute, x, y):
        self.f, self.Pxx = signal.welch(x, self.sfreq, self.window, nperseg=128)
        self.Pyy = signal.welch(y, self.sfreq, self.window, nperseg=128)[1]
        self.Pxy = signal.csd(x, y, self.sfreq, self.window, nperseg=128)[1]

        if self.coh is True:
            self.coh_val = np.abs(self.Pxy**2) / (self.Pxx * self.Pyy)
        if self.icoh is True:
            self.icoh_val = np.array(self.Pxy / (self.Pxx * self.Pyy)).imag

        for coh_idx, coh_type in enumerate([self.coh, self.icoh]):
            if coh_type is True:
                if coh_idx == 0:
                    coh_val = self.coh_val
                    coh_name = "coh"
                else:
                    coh_val = self.icoh_val
                    coh_name = "icoh"

            for idx, fband in enumerate(self.fbands):
                if self.features_coh["mean_fband"] is True:
                    feature_calc = np.mean(
                        coh_val[
                            np.bitwise_and(self.f > fband[0], self.f < fband[1])
                        ]
                    )
                    feature_name = "_".join(
                        [
                            coh_name,
                            self.ch_1,
                            "to",
                            self.ch_2,
                            "mean_fband",
                            self.fband_names[idx],
                        ]
                    )
                    features_compute[feature_name] = feature_calc
                if self.features_coh["max_fband"] is True:
                    feature_calc = np.max(
                        coh_val[
                            np.bitwise_and(self.f > fband[0], self.f < fband[1])
                        ]
                    )
                    feature_name = "_".join(
                        [
                            coh_name,
                            self.ch_1,
                            "to",
                            self.ch_2,
                            "max_fband",
                            self.fband_names[idx],
                        ]
                    )
                    features_compute[feature_name] = feature_calc
            if self.features_coh["max_allfbands"] is True:
                feature_calc = self.f[np.argmax(coh_val)]
                feature_name = "_".join(
                    [
                        coh_name,
                        self.ch_1,
                        "to",
                        self.ch_2,
                        "max_allfbands",
                        self.fband_names[idx],
                    ]
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
            fband_names = self.s["coherence"]["frequency_bands"]
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
                    "hann",
                    fband_specs,
                    fband_names,
                    ch_1_name,
                    ch_2_name,
                    ch_1_idx,
                    ch_2_idx,
                    self.s["coherence"]["method"]["coh"],
                    self.s["coherence"]["method"]["icoh"],
                    self.s["coherence"]["features"],
                )
            )

    @staticmethod
    def test_settings(
        s: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):

        assert (
            len(s["coherence"]["frequency_bands"]) > 0
        ), "coherence frequency_bands list needs to specify at least one frequency band"
        assert (ch_coh in ch_names for ch_coh in s["coherence"]["channels"]), (
            f"coherence selected channels don't match the ones in nm_channels"
            f"ch_names: {ch_names} settings['coherence']['channels']: {s['coherence']['channels']}"
        )

        assert (
            f_band_coh in s["frequency_ranges_hz"]
            for f_band_coh in s["coherence"]["frequency_bands"]
        ), (
            "coherence selected frequency bands don't match the ones"
            "specified in s['frequency_ranges_hz']"
            f"coherence frequency bands: {s['coherence']['frequency_bands']}"
            f"specified frequency_ranges_hz: {s['frequency_ranges_hz']}"
        )

        assert (
            s["frequency_ranges_hz"][fb][0] < sfreq / 2
            and s["frequency_ranges_hz"][fb][1] < sfreq / 2
            for fb in s["coherence"]["frequency_bands"]
        ), (
            "the coherence frequency band ranges need to be smaller than the nyquist frequency"
            f"got sfreq = {sfreq} and fband ranges {s['coherence']['frequency_bands']}"
        )

        if sum(list(s["coherence"]["method"].values())) == 0:
            warnings.warn(
                "feature coherence enabled, but no coherence['method'] selected"
            )

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:
        for coh_obj in self.coherence_objects:
            features_compute = coh_obj.get_coh(
                features_compute,
                data[coh_obj.ch_1_idx, :],
                data[coh_obj.ch_2_idx, :],
            )

        return features_compute
