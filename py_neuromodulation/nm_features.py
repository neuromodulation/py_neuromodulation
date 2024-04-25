from importlib import import_module
from typing import Protocol
from collections.abc import Iterable
import numpy as np


class NMFeature(Protocol):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: int | float
    ) -> None:
        """Method to check passed settings"""

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        """Method to check passed settings"""

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        """
        Feature calculation method. Each method needs to loop through all channels

        Parameters
        ----------
        data : np.ndarray
            (channels, time)
        features_compute : dict
        ch_names : Iterable[str]

        Returns
        -------
        dict
        """

FEATURE_DICT : dict[str, tuple[str,str]] = {
    "raw_hjorth": ("py_neuromodulation.nm_hjorth_raw", "Hjorth") ,
    "return_raw": ("py_neuromodulation.nm_hjorth_raw", "Raw") ,
    "bandpass_filter": ("py_neuromodulation.nm_oscillatory", "BandPower") ,
    "stft": ("py_neuromodulation.nm_oscillatory", "STFT") ,
    "fft": ("py_neuromodulation.nm_oscillatory", "FFT") ,
    "welch": ("py_neuromodulation.nm_oscillatory", "Welch") ,
    "sharpwave_analysis": ("py_neuromodulation.nm_sharpwaves", "SharpwaveAnalyzer") ,
    "fooof": ("py_neuromodulation.nm_fooof", "FooofAnalyzer") ,
    "nolds": ("py_neuromodulation.nm_nolds", "Nolds") ,
    "coherence": ("py_neuromodulation.nm_coherence", "NM_Coherence") ,
    "bursts": ("py_neuromodulation.nm_bursts", "Burst") ,
    "linelength": ("py_neuromodulation.nm_linelength", "LineLength") ,
    "mne_connectivity": ("py_neuromodulation.nm_mne_connectivity", "MNEConnectivity") ,
    "bispectrum": ("py_neuromodulation.nm_bispectra", "Bispectra")
}
     
             
class Features:
    """Class for calculating features.p"""

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

        self.features: list[NMFeature]  = []
        
        for feature_name, feature_enabled in s["features"].items():
            if feature_name not in FEATURE_DICT:
                raise ValueError(f"Unknown feature found. Got: {feature_name}.")

            if feature_enabled:
                feature_module = import_module(FEATURE_DICT[feature_name][0])
                feature_classobj = getattr(feature_module, FEATURE_DICT[feature_name][1])
                feature_classobj.test_settings(s, ch_names, sfreq)
                feature_instance = feature_classobj(s, ch_names, sfreq)
                self.features.append(feature_instance)


    def register_new_feature(self, feature: NMFeature) -> None:
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
