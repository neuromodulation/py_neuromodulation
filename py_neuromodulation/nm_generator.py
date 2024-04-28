from collections.abc import Iterator
import numpy as np


def raw_data_generator(
    data: np.ndarray,
    settings: dict,
    sfreq: float,
) -> Iterator[np.ndarray]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: float
    Returns
    -------
        np.array: new batch for run function of full segment length shape
    """
    sfreq_new = settings["sampling_rate_features_hz"]
    offset_time = settings["segment_length_features_ms"]
    offset_start = offset_time / 1000 * sfreq

    ratio_samples_features = sfreq / sfreq_new

    ratio_counter = 0
    for cnt in range(
        data.shape[1] + 1
    ):  # shape + 1 guarantees that the last sample is also included
        if (cnt - offset_start) >= ratio_samples_features * ratio_counter:
            ratio_counter += 1

            yield data[:, np.floor(cnt - offset_start).astype(int) : cnt]
