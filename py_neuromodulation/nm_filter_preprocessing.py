import numpy as np

from pydantic import Field
from typing import TYPE_CHECKING

from py_neuromodulation.nm_types import FeatureSelector, FrequencyRange
from py_neuromodulation.nm_preprocessing import NMPreprocessor

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


FILTER_SETTINGS_MAP = {
    "bandstop_filter": "bandstop_filter_settings",
    "bandpass_filter": "bandpass_filter_settings",
    "lowpass_filter": "lowpass_filter_cutoff_hz",
    "highpass_filter": "highpass_filter_cutoff_hz",
}


class FilterSettings(FeatureSelector):
    bandstop_filter: bool = True
    bandpass_filter: bool = True
    lowpass_filter: bool = True
    highpass_filter: bool = True

    bandstop_filter_settings: FrequencyRange = FrequencyRange(100, 160)
    bandpass_filter_settings: FrequencyRange = FrequencyRange(2, 200)
    lowpass_filter_cutoff_hz: float = Field(default=200)
    highpass_filter_cutoff_hz: float = Field(default=3)

    def get_filter_tuple(self, filter_name) -> tuple[float | None, float | None]:
        filter_value = self[FILTER_SETTINGS_MAP[filter_name]]

        match filter_name:
            case "bandstop_filter":
                return (filter_value.frequency_high_hz, filter_value.frequency_low_hz)
            case "bandpass_filter":
                return (filter_value.frequency_low_hz, filter_value.frequency_high_hz)
            case "lowpass_filter":
                return (None, filter_value)
            case "highpass_filter":
                return (filter_value, None)
            case _:
                raise ValueError(
                    "Filter name must be one of 'bandstop_filter', 'lowpass_filter', "
                    "'highpass_filter', 'bandpass_filter'"
                )


class PreprocessingFilter(NMPreprocessor):
    def __init__(self, settings: "NMSettings", sfreq: float) -> None:
        from py_neuromodulation.nm_filter import MNEFilter

        self.filters: list[MNEFilter] = [
            MNEFilter(
                f_ranges=[settings.preprocessing_filter.get_filter_tuple(filter_name)],  # type: ignore
                sfreq=sfreq,
                filter_length=sfreq - 1,
                verbose=False,
            )
            for filter_name in settings.preprocessing_filter.get_enabled()
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
