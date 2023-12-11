import numpy as np

from py_neuromodulation import (
    nm_hjorth_raw,
    nm_mne_connectivity,
    nm_sharpwaves,
    nm_coherence,
    nm_fooof,
    nm_nolds,
    nm_features_abc,
    nm_oscillatory,
    nm_bursts,
    nm_linelength,
    nm_bispectra,
)


class Features:
    """Class for calculating features."""

    features: list[nm_features_abc.Feature] = []

    def __init__(
        self, s: dict, ch_names: list[str], sfreq: int | float
    ) -> None:
        """_summary_

        Parameters
        ----------
        s : dict
            _description_
        ch_names : list[str]
            _description_
        sfreq : int | float
            _description_

        Raises
        ------
        ValueError
            _description_
        """

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
                case "welch":
                    FeatureClass = nm_oscillatory.Welch
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
                    FeatureClass = nm_mne_connectivity.MNEConnectivity
                case "bispectrum":
                    FeatureClass = nm_bispectra.Bispectra
                case _:
                    raise ValueError(f"Unknown feature found. Got: {feature}.")

            FeatureClass.test_settings(s, ch_names, sfreq)
            f_obj = FeatureClass(s, ch_names, sfreq)
            self.features.append(f_obj)

    def register_new_feature(self, feature: nm_features_abc.Feature) -> None:
        """Register new feature.

        Parameters
        ----------
        feature : nm_features_abc.Feature
            New feature to add to feature list
        """
        self.features.append(feature)

    def estimate_features(self, data: np.ndarray) -> dict:
        """Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.

        Parameters
        ----------
        data (np array) : (channels, time)

        Returns
        -------
        dat (dict): naming convention : channel_method_feature_(f_band)
        """

        features_compute = {}

        for feature in self.features:
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute
