from collections.abc import Iterator
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


def raw_data_generator(
    data: np.ndarray,
    settings: "NMSettings",
    sfreq: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with settings.sampling_rate_features_hz

    Arguments
    ---------
        data (np array): shape (channels, time)
        settings (nm_settings.NMSettings): settings object
        sfreq (float): sampling frequency of the data

    Returns
    -------
        np.array: 1D array of time stamps
        np.array: new batch for run function of full segment length shape
    """
    sfreq_new = settings.sampling_rate_features_hz
    offset_time = settings.segment_length_features_ms
    offset_start = offset_time / 1000 * sfreq

    ratio_samples_features = sfreq / sfreq_new

    ratio_counter = 0
    for cnt in range(
        data.shape[1] + 1
    ):  # shape + 1 guarantees that the last sample is also included
        if (cnt - offset_start) >= ratio_samples_features * ratio_counter:
            ratio_counter += 1

            yield (
                np.arange(cnt - offset_start, cnt) / sfreq,
                data[:, np.floor(cnt - offset_start).astype(int) : cnt],
            )
