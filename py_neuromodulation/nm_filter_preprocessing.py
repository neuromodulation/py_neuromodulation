import numpy as np

from py_neuromodulation import nm_filter


class PreprocessingFilter:

    def __init__(
        self, settings: dict, sfreq: int | float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.filters = []

        if self.s["preprocessing_filter"]["bandstop_filter"] is True:
            self.filters.append(
                nm_filter.MNEFilter(
                    f_ranges=[
                        self.s["preprocessing_filter"][
                            "bandstop_filter_settings"
                        ]["frequency_low_hz"],
                        self.s["preprocessing_filter"][
                            "bandstop_filter_settings"
                        ]["frequency_high_hz"],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )

        if self.s["preprocessing_filter"]["bandpass_filter"] is True:
            self.filters.append(
                nm_filter.MNEFilter(
                    f_ranges=[
                        self.s["preprocessing_filter"][
                            "bandpass_filter_settings"
                        ]["frequency_low_hz"],
                        self.s["preprocessing_filter"][
                            "bandpass_filter_settings"
                        ]["frequency_high_hz"],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )
        if self.s["preprocessing_filter"]["lowpass_filter"] is True:
            self.filters.append(
                nm_filter.MNEFilter(
                    f_ranges=[
                        None,
                        self.s["preprocessing_filter"][
                            "lowpass_filter_settings"
                        ]["frequency_cutoff_hz"],
                    ],
                    sfreq=self.sfreq,
                    filter_length=self.sfreq - 1,
                    verbose=False,
                )
            )
        if self.s["preprocessing_filter"]["highpass_filter"] is True:
            self.filters.append(
                nm_filter.MNEFilter(
                    f_ranges=[
                        self.s["preprocessing_filter"][
                            "highpass_filter_settings"
                        ]["frequency_cutoff_hz"],
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
            data = filter.filter_data(data)

        return data
