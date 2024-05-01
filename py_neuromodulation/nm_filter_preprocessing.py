import numpy as np

from py_neuromodulation.nm_filter import MNEFilter


class PreprocessingFilter:
    def __init__(self, settings: dict, sfreq: float) -> None:
        self.settings = settings
        self.sfreq = sfreq
        self.filters = []

        if self.settings["preprocessing_filter"]["bandstop_filter"]:
            self.filters.append(
                MNEFilter(
                    f_ranges=[
                        self.settings["preprocessing_filter"]["bandstop_filter_settings"][
                            "frequency_high_hz"
                        ],
                        self.settings["preprocessing_filter"]["bandstop_filter_settings"][
                            "frequency_low_hz"
                        ],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )

        if self.settings["preprocessing_filter"]["bandpass_filter"]:
            self.filters.append(
                MNEFilter(
                    f_ranges=[
                        self.settings["preprocessing_filter"]["bandpass_filter_settings"][
                            "frequency_low_hz"
                        ],
                        self.settings["preprocessing_filter"]["bandpass_filter_settings"][
                            "frequency_high_hz"
                        ],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )
        if self.settings["preprocessing_filter"]["lowpass_filter"]:
            self.filters.append(
                MNEFilter(
                    f_ranges=[
                        None,
                        self.settings["preprocessing_filter"]["lowpass_filter_settings"][
                            "frequency_cutoff_hz"
                        ],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )
        if self.settings["preprocessing_filter"]["highpass_filter"]:
            self.filters.append(
                MNEFilter(
                    f_ranges=[
                        self.settings["preprocessing_filter"]["highpass_filter_settings"][
                            "frequency_cutoff_hz"
                        ],
                        None,
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
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
