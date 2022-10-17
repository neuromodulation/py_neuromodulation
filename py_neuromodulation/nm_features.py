# from multiprocessing import Process, Manager
import numpy as np
from typing import Iterable

from py_neuromodulation import (
    nm_hjorth_raw,
    nm_sharpwaves,
    nm_coherence,
    nm_fooof,
    nm_nolds,
    nm_features_abc,
    nm_oscillatory,
    nm_bursts,
    nm_linelength,
    nm_mne_connectiviy
)


class Features:

    sfreq: float
    features: Iterable[nm_features_abc.Feature] = []

    def __init__(self, s, ch_names, sfreq) -> None:

        if s["preprocessing"]["raw_resampling"] is True:
            sfreq = s["raw_resampling_settings"]["resample_freq_hz"]

        self.sfreq = sfreq

        for feature in s["features"]:
            if s["features"][feature] is False:
                continue
            match feature:
                case "raw_hjorth":
                    self.features.append(
                        nm_hjorth_raw.Hjorth(s, ch_names, sfreq)
                    )
                case "return_raw":
                    self.features.append(nm_hjorth_raw.Raw(s, ch_names, sfreq))
                case "bandpass_filter":
                    self.features.append(
                        nm_oscillatory.BandPower(s, ch_names, sfreq)
                    )
                case "stft":
                    self.features.append(
                        nm_oscillatory.STFT(s, ch_names, sfreq)
                    )
                case "fft":
                    self.features.append(nm_oscillatory.FFT(s, ch_names, sfreq))
                case "sharpwave_analysis":
                    self.features.append(
                        nm_sharpwaves.SharpwaveAnalyzer(s, ch_names, sfreq)
                    )
                case "fooof":
                    self.features.append(
                        nm_fooof.FooofAnalyzer(s, ch_names, sfreq)
                    )
                case "nolds":
                    self.features.append(nm_nolds.Nolds(s, ch_names, sfreq))
                case "coherence":
                    self.features.append(
                        nm_coherence.NM_Coherence(s, ch_names, sfreq)
                    )
                case "bursts":
                    self.features.append(nm_bursts.Burst(s, ch_names, sfreq))
                case "linelength":
                    self.features.append(nm_linelength.LineLengh(s, ch_names, sfreq))
                case "mne_connectiviy":
                    self.features.append(nm_mne_connectiviy.MNEConnectivity(s, ch_names, sfreq))

    def estimate_features(self, data) -> dict:
        """Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.

        Parameters
        ----------
        data (np array) : (channels, time)

        Returns
        -------
        dat (dict) with naming convention:
            channel_method_feature_(f_band)
        """

        features_compute = {}

        for feature in self.features:
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute
