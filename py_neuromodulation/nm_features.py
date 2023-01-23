"""Module for calculating features."""
import numpy as np

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
    nm_mne_connectiviy,
)


class Features:

    features: list[nm_features_abc.Feature] = []

    def __init__(
        self, s: dict, ch_names: list[str], sfreq: int | float
    ) -> None:
        """Class for calculating features."""
        self.features = []

        for feature in s["features"]:
            if s["features"][feature] is False:
                continue
            match feature:
                case "raw_hjorth":
                    FeatureClass = nm_hjorth_raw.Hjorth
                case "return_raw":
                    FeatureClass = nm_hjorth_raw.Raw
                case "bandpass_filter":
                    FeatureClass = nm_oscillatory.BandPower
                case "stft":
                    FeatureClass = nm_oscillatory.STFT
                case "fft":
                    FeatureClass = nm_oscillatory.FFT
                case "sharpwave_analysis":
                    FeatureClass = nm_sharpwaves.SharpwaveAnalyzer
                case "fooof":
                    FeatureClass = nm_fooof.FooofAnalyzer
                case "nolds":
                    FeatureClass = nm_nolds.Nolds
                case "coherence":
                    FeatureClass = nm_coherence.NM_Coherence
                case "bursts":
                    FeatureClass = nm_bursts.Burst
                case "linelength":
                    FeatureClass = nm_linelength.LineLength
                case "mne_connectivity":
                    FeatureClass = nm_mne_connectiviy.MNEConnectivity
                case _:
                    raise ValueError(f"Unknown feature found. Got: {feature}.")

            FeatureClass.test_settings(s, ch_names, sfreq)
            f_obj = FeatureClass(s, ch_names, sfreq)
            self.features.append(f_obj)

    def estimate_features(self, data: np.ndarray) -> dict:
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
