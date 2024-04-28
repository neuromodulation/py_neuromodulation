import numpy as np

from py_neuromodulation.nm_hjorth_raw import Hjorth, Raw
from py_neuromodulation.nm_oscillatory import BandPower, STFT, FFT, Welch
from py_neuromodulation.nm_sharpwaves import SharpwaveAnalyzer
from py_neuromodulation.nm_fooof import FooofAnalyzer
from py_neuromodulation.nm_nolds import Nolds
from py_neuromodulation.nm_coherence import NM_Coherence
from py_neuromodulation.nm_bursts import Burst
from py_neuromodulation.nm_linelength import LineLength
from py_neuromodulation.nm_mne_connectivity import MNEConnectivity
from py_neuromodulation.nm_bispectra import Bispectra

from py_neuromodulation.nm_features_abc import Feature
class Features:
    """Class for calculating features.p"""

    # features: list[nm_features_abc.Feature] = []

    def __init__(
        self, s: dict, ch_names: list[str], sfreq: float
    ) -> None:
        """_summary_

        Parameters
        ----------
        s : dict
            _description_
        ch_names : list[str]
            _description_
        sfreq : float
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        self.features: list[Feature]  = []

        feature: Feature
        for feature in s["features"]:
            if not s["features"][feature]:
                continue
            match feature:
                case "raw_hjorth":
                    FeatureClass = Hjorth
                case "return_raw":
                    FeatureClass = Raw
                case "bandpass_filter":
                    FeatureClass = BandPower
                case "stft":
                    FeatureClass = STFT
                case "fft":
                    FeatureClass = FFT
                case "welch":
                    FeatureClass = Welch
                case "sharpwave_analysis":
                    FeatureClass = SharpwaveAnalyzer
                case "fooof":
                    FeatureClass = FooofAnalyzer
                case "nolds":
                    FeatureClass = Nolds
                case "coherence":
                    FeatureClass = NM_Coherence
                case "bursts":
                    FeatureClass = Burst
                case "linelength":
                    FeatureClass = LineLength
                case "mne_connectivity":
                    FeatureClass = MNEConnectivity
                case "bispectrum":
                    FeatureClass = Bispectra
                case _:
                    raise ValueError(f"Unknown feature found. Got: {feature}.")

            FeatureClass.test_settings(s, ch_names, sfreq)
            f_obj = FeatureClass(s, ch_names, sfreq)
            self.features.append(f_obj)

    def register_new_feature(self, feature: Feature) -> None:
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

        features_compute : dict = {}

        for feature in self.features:
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute
