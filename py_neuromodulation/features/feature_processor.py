from typing import Type, TYPE_CHECKING

from py_neuromodulation.utils.types import NMFeature, FeatureName

if TYPE_CHECKING:
    import numpy as np
    from py_neuromodulation import NMSettings


FEATURE_DICT: dict[FeatureName | str, str] = {
    "raw_hjorth": "Hjorth",
    "return_raw": "Raw",
    "bandpass_filter": "BandPower",
    "stft": "STFT",
    "fft": "FFT",
    "welch": "Welch",
    "sharpwave_analysis": "SharpwaveAnalyzer",
    "fooof": "FooofAnalyzer",
    "nolds": "Nolds",
    "coherence": "Coherence",
    "bursts": "Bursts",
    "linelength": "LineLength",
    "mne_connectivity": "MNEConnectivity",
    "bispectrum": "Bispectra",
}


class FeatureProcessors:
    """Class for storing NMFeature objects and calculating features during processing"""

    def __init__(
        self, settings: "NMSettings", ch_names: list[str], sfreq: float
    ) -> None:
        """Initialize FeatureProcessors object with settings, channel names and sampling frequency.

        Args:
            settings (NMSettings): PyNM settings object
            ch_names (list[str]): list of channel names
            sfreq (float): sampling frequency in Hz
        """
        from py_neuromodulation import user_features
        from importlib import import_module

        # Accept 'str' for custom features
        self.features: dict[FeatureName | str, NMFeature] = {
            feature_name: getattr(
                import_module("py_neuromodulation.features"), FEATURE_DICT[feature_name]
            )(settings, ch_names, sfreq)
            for feature_name in settings.features.get_enabled()
        }

        for feature_name, feature in user_features.items():
            self.features[feature_name] = feature(settings, ch_names, sfreq)

    def register_new_feature(self, feature_name: str, feature: NMFeature) -> None:
        """Register new feature.

        Parameters
        ----------
        feature : features_abc.Feature
            New feature to add to feature list
        """
        self.features[feature_name] = feature  # type: ignore

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

        feature_results: dict = {}

        for feature in self.features.values():
            feature_results.update(feature.calc_feature(data))

        return feature_results

    def get_feature(self, fname: FeatureName) -> NMFeature:
        return self.features[fname]


def add_custom_feature(feature_name: str, new_feature: Type[NMFeature]):
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
    from py_neuromodulation import NMSettings

    user_features[feature_name] = new_feature
    NMSettings._add_feature(feature_name)


def remove_custom_feature(feature_name: str):
    """Remove a custom feature from the dictionary of user-defined features.

    Args:
        feature_name (str): Name of the feature to remove
    """
    from py_neuromodulation import user_features
    from py_neuromodulation import NMSettings

    user_features.pop(feature_name)
    NMSettings._remove_feature(feature_name)
