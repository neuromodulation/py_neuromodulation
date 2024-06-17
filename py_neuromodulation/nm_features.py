from typing import Protocol, Type, runtime_checkable, TYPE_CHECKING, TypeVar
from collections.abc import Sequence

if TYPE_CHECKING:
    import numpy as np
    from nm_settings import NMSettings

from py_neuromodulation.nm_types import ImportDetails, get_class, FeatureName

@runtime_checkable
class NMFeature(Protocol):
    def __init__(
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: int | float
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


NMFeatureType = TypeVar("NMFeatureType", bound=NMFeature)

FEATURE_DICT: dict[FeatureName | str, ImportDetails] = {
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


class FeatureProcessors:
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
        from py_neuromodulation import user_features
    
        # Accept 'str' for custom features
        self.features: dict[FeatureName | str, NMFeature] = {
            feature_name: get_class(FEATURE_DICT[feature_name])(
                settings, ch_names, sfreq
            )
            for feature_name in settings.features.get_enabled()
        }
        
        for feature_name, feature in user_features.items():
            self.features[feature_name] = feature(settings, ch_names, sfreq)

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


def AddCustomFeature(feature_name: str, new_feature: Type[NMFeature]):
    """Add a custom feature to the dictionary of user-defined features.
        The feature will be automatically enabled in the settings,
        and computed when the Stream.run() method is called.

    Args:
        feature_name (str): A name for the feature that will be used to
        enable/disable the feature in settings and to store the feature
        class instance in the DataProcessor

        new_feature (NMFeature): Class that implements the user-defined
        feature. It should implement the NMSettings protocol (defined
        in this file).
    """
    from py_neuromodulation import user_features
    from py_neuromodulation.nm_settings import NMSettings

    user_features[feature_name] = new_feature
    NMSettings._add_feature(feature_name)


def RemoveCustomFeature(feature_name: str):
    """Remove a custom feature from the dictionary of user-defined features.

    Args:
        feature_name (str): Name of the feature to remove
    """
    from py_neuromodulation import user_features

    user_features.pop(feature_name)
    NMSettings._remove_feature(feature_name)
