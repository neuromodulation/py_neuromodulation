import numpy as np

from typing import TYPE_CHECKING

from py_neuromodulation.utils.types import BoolSelector, FrequencyRange, NMPreprocessor
from py_neuromodulation.utils.pydantic_extensions import NMField

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class FilterSettings(BoolSelector):
    bandstop_filter: bool = True
    bandpass_filter: bool = True
    lowpass_filter: bool = True
    highpass_filter: bool = True

    bandstop_filter_settings: FrequencyRange = FrequencyRange(100, 160)
    bandpass_filter_settings: FrequencyRange = FrequencyRange(2, 200)
    lowpass_filter_cutoff_hz: float = NMField(
        default=200, gt=0, custom_metadata={"unit": "Hz"}
    )
    highpass_filter_cutoff_hz: float = NMField(
        default=3, gt=0, custom_metadata={"unit": "Hz"}
    )

    def get_filter_tuple(self, filter_name) -> FrequencyRange:
        match filter_name:
            case "bandstop_filter":
                return self.bandstop_filter_settings
            case "bandpass_filter":
                return self.bandpass_filter_settings
            case "lowpass_filter":
                return FrequencyRange(None, self.lowpass_filter_cutoff_hz)
            case "highpass_filter":
                return FrequencyRange(self.highpass_filter_cutoff_hz, None)
            case _:
                raise ValueError(
                    "Filter name must be one of 'bandstop_filter', 'lowpass_filter', "
                    "'highpass_filter', 'bandpass_filter'"
                )


class PreprocessingFilter(NMPreprocessor):
    def __init__(self, settings: "NMSettings", sfreq: float) -> None:
        from py_neuromodulation.filter import MNEFilter


        self.filters: list[MNEFilter] = []
        for filter_name in settings.preprocessing_filter.get_enabled():
            if filter_name != "lowpass_filter" and filter_name != "highpass_filter":
                self.filters += [
                    MNEFilter(
                        f_ranges=[settings.preprocessing_filter.get_filter_tuple(filter_name)],  # type: ignore
                        sfreq=sfreq,
                        filter_length=sfreq - 1,
                        verbose=False,
                    )
                ]

        if "lowpass_filter" in settings.preprocessing_filter.get_enabled():
            self.filters.append(
                MNEFilter(
                    f_ranges=[[None, settings.preprocessing_filter.lowpass_filter_cutoff_hz]],  # type: ignore
                    sfreq=sfreq,
                    filter_length=sfreq - 1,
                    verbose=False,
                )
            )
        if "highpass_filter" in settings.preprocessing_filter.get_enabled():
            self.filters.append(
                MNEFilter(
                    f_ranges=[[settings.preprocessing_filter.highpass_filter_cutoff_hz, None]],  # type: ignore
                    sfreq=sfreq,
                    filter_length=sfreq - 1,
                    verbose=False,
                )
            )

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
