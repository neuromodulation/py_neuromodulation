import numpy as np
from py_neuromodulation.stream.settings import NMSettings


class RawDataGenerator:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with settings.sampling_rate_features_hz
    """

    def __init__(
        self,
        data: np.ndarray,
        settings: NMSettings,
        sfreq: float,
    ) -> None:
        """
        Arguments
        ---------
            data (np array): shape (channels, time)
            settings (settings.NMSettings): settings object
            sfreq (float): sampling frequency of the data

        Returns
        -------
            np.array: 1D array of time stamps
            np.array: new batch for run function of full segment length shape
        """
        self._idx = None

        self.sfreq = sfreq
        self.data = data

        sfreq_new = settings.sampling_rate_features_hz
        offset_time = settings.segment_length_features_ms
        self.offset_start = offset_time / 1000 * sfreq

        self.ratio_samples_features = sfreq / sfreq_new

        self.ratio_counter = 0

        self.n_samples = range(
            data.shape[1] + 1
        )  # shape + 1 guarantees that the last sample is also included

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx is None:
            self._idx = 0

        try:
            return (
                np.arange(cnt - self.offset_start, cnt) / self.sfreq,
                self.data[:, np.floor(cnt - self.offset_start).astype(int) : cnt],
            )

        except IndexError:
            raise StopIteration
