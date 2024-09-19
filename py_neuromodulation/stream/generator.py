import numpy as np


class RawDataGenerator:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with settings.sampling_rate_features_hz
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        sampling_rate_features_hz: float,
        segment_length_features_ms: float,
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
        self.batch_counter: int = 0  # counter for the batches

        self.data = data
        self.sfreq = sfreq
        # Width, in data points, of the moving window used to calculate features
        self.segment_length = segment_length_features_ms / 1000 * sfreq
        # Ratio of the sampling frequency of the input data to the sampling frequency
        self.stride = sfreq / sampling_rate_features_hz

    def __iter__(self):
        return self

    def __next__(self):
        start = self.stride * self.batch_counter
        end = start + self.segment_length

        self.batch_counter += 1

        start_idx = int(start)
        end_idx = int(end)

        if end_idx > self.data.shape[1]:
            raise StopIteration

        return np.arange(start, end) / self.sfreq, self.data[:, start_idx:end_idx]
