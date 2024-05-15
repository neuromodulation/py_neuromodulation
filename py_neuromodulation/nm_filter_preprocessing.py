import numpy as np

from pydantic.dataclasses import dataclass
from pydantic import Field

from py_neuromodulation.nm_preprocessing import NMPreprocessor
from py_neuromodulation.nm_settings import NMSettings
from py_neuromodulation.nm_types import FeatureSelector, FrequencyRange


@dataclass
class FrequencyLowpass(FrequencyRange):
    frequency_low_hz: float = float("nan")
    frequency_high_hz: float = Field(default=None, alias="frequency_cutoff_hz")


@dataclass
class FrequencyHighpass(FrequencyRange):
    frequency_low_hz: float = Field(default=None, alias="frequency_cutoff_hz")
    frequency_high_hz: float = float("nan")


@dataclass
class FilterSettings(FeatureSelector):
    bandstop_filter: bool = True
    lowpass_filter: bool = True
    highpass_filter: bool = True
    bandpass_filter: bool = True

    bandstop_filter_settings: FrequencyRange = FrequencyRange(100, 160)
    bandpass_filter_settings: FrequencyRange = FrequencyRange(2, 200)
    lowpass_filter_settings: FrequencyLowpass = FrequencyLowpass(float("nan"), 200)
    highpass_filter_settings: FrequencyHighpass = FrequencyHighpass(3, float("nan"))

    def get_enabled(self):
        return [
            name
            for name in (
                "bandstop_filter",
                "lowpass_filter",
                "highpass_filter",
                "bandpass_filter",
            )
            if getattr(self, name)
        ]

    def get_filter_settings(self, name):
        return getattr(self, name + "_settings")


class PreprocessingFilter(NMPreprocessor):
    def __init__(self, settings: NMSettings, sfreq: float) -> None:
        from py_neuromodulation.nm_filter import MNEFilter

        self.filters = [
            MNEFilter(
                f_ranges=[settings.preprocessing_filter.get_filter_settings(filter)[0]],
                sfreq=sfreq,
                filter_length=sfreq - 1,
                verbose=False,
            )
            for filter in settings.preprocessing_filter.get_enabled()
        ]

    def process(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data according to the initialized list of PreprocessingFilter objects

        Args:
            data (numpy ndarray) :
                shape(n_channels, n_samples) - data to be preprocessed.

        Returns:
            preprocessed_data (numpy ndarray):
            shape(n_channels, n_samples) - preprocessed data
        """

        for filter in self.filters:
            data = filter.filter_data(data if len(data.shape) == 2 else data[:, 0, :])
        return data if len(data.shape) == 2 else data[:, 0, :]
