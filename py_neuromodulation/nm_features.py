from typing import Protocol, TYPE_CHECKING, cast
from collections.abc import Iterable

if TYPE_CHECKING:
    import numpy as np
    from nm_settings import NMSettings

from py_neuromodulation.nm_types import ImportDetails, get_class, FeatureName


class NMFeature(Protocol):
    def __init__(
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: int | float
    ) -> None: ...

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


FEATURE_DICT: dict[FeatureName, ImportDetails] = {
    "raw_hjorth": ImportDetails("nm_hjorth_raw", "Hjorth"),
    "return_raw": ImportDetails("nm_hjorth_raw", "Raw"),
    "bandpass_filter": ImportDetails("nm_oscillatory", "BandPower"),
    "stft": ImportDetails("nm_oscillatory", "STFT"),
    "fft": ImportDetails("nm_oscillatory", "FFT"),
    "welch": ImportDetails("nm_oscillatory", "Welch"),
    "sharpwave_analysis": ImportDetails("nm_sharpwaves", "SharpwaveAnalyzer"),
    "fooof": ImportDetails("nm_fooof", "FooofAnalyzer"),
    "nolds": ImportDetails("nm_nolds", "Nolds"),
    "coherence": ImportDetails("nm_coherence", "NMCoherence"),
    "bursts": ImportDetails("nm_bursts", "Burst"),
    "linelength": ImportDetails("nm_linelength", "LineLength"),
    "mne_connectivity": ImportDetails("nm_mne_connectivity", "MNEConnectivity"),
    "bispectrum": ImportDetails("nm_bispectra", "Bispectra"),
}


class Features:
    """Class for calculating features.p"""

    def __init__(
        self, settings: "NMSettings", ch_names: list[str], sfreq: float
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

        # Accept 'str' for custom features
        self.features: dict[FeatureName | str, NMFeature] = {
            feature_name: get_class(FEATURE_DICT[feature_name])(
                settings, ch_names, sfreq
            )
            for feature_name, feature_enabled in settings.features.items()
            if feature_enabled
        }

    def register_new_feature(self, feature_name: str, feature: NMFeature) -> None:
        """Register new feature.

        Parameters
        ----------
        feature : nm_features_abc.Feature
            New feature to add to feature list
        """
        self.features[feature_name] = feature  # type: ignore

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

        for feature in self.features.values():
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute

    def get_feature(self, fname: FeatureName) -> NMFeature:
        return self.features[fname]
