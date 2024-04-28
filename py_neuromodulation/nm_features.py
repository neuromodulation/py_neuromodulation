from typing import Protocol, TYPE_CHECKING
from collections.abc import Iterable

if TYPE_CHECKING:
    import numpy as np

from py_neuromodulation.nm_types import ImportDetails, get_class


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

    def calc_feature(self, data: "np.ndarray", features_compute: dict) -> dict:
        """
        Feature calculation method. Each method needs to loop through all channels

        Parameters
        ----------
        data : 'np.ndarray'
            (channels, time)
        features_compute : dict
        ch_names : Iterable[str]

        Returns
        -------
        dict
        """
        ...


FEATURE_DICT: dict[str, ImportDetails] = {
    "raw_hjorth": ImportDetails("py_neuromodulation.nm_hjorth_raw", "Hjorth"),
    "return_raw": ImportDetails("py_neuromodulation.nm_hjorth_raw", "Raw"),
    "bandpass_filter": ImportDetails("py_neuromodulation.nm_oscillatory", "BandPower"),
    "stft": ImportDetails("py_neuromodulation.nm_oscillatory", "STFT"),
    "fft": ImportDetails("py_neuromodulation.nm_oscillatory", "FFT"),
    "welch": ImportDetails("py_neuromodulation.nm_oscillatory", "Welch"),
    "sharpwave_analysis": ImportDetails(
        "py_neuromodulation.nm_sharpwaves", "SharpwaveAnalyzer"
    ),
    "fooof": ImportDetails("py_neuromodulation.nm_fooof", "FooofAnalyzer"),
    "nolds": ImportDetails("py_neuromodulation.nm_nolds", "Nolds"),
    "coherence": ImportDetails("py_neuromodulation.nm_coherence", "NM_Coherence"),
    "bursts": ImportDetails("py_neuromodulation.nm_bursts", "Burst"),
    "linelength": ImportDetails("py_neuromodulation.nm_linelength", "LineLength"),
    "mne_connectivity": ImportDetails(
        "py_neuromodulation.nm_mne_connectivity", "MNEConnectivity"
    ),
    "bispectrum": ImportDetails("py_neuromodulation.nm_bispectra", "Bispectra"),
}


class Features:
    """Class for calculating features.p"""

    def __init__(self, settings: dict, ch_names: list[str], sfreq: float) -> None:
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

        for feature_name in settings["features"]:
            if feature_name not in FEATURE_DICT:
                raise ValueError(f"Unknown feature found. Got: {feature_name}.")

        self.features: list[NMFeature] = [
            get_class(FEATURE_DICT[feature_name])(settings, ch_names, sfreq)
            for feature_name, feature_enabled in settings["features"].items()
            if feature_enabled
        ]
        
        for feature in self.features:
            feature.test_settings(settings, ch_names, sfreq)

    def register_new_feature(self, feature: NMFeature) -> None:
        """Register new feature.

        Parameters
        ----------
        feature : nm_features_abc.Feature
            New feature to add to feature list
        """
        self.features.append(feature)

    def estimate_features(self, data: "np.ndarray") -> dict:
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

        features_compute: dict = {}

        for feature in self.features:
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute
