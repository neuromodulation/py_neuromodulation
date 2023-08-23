from typing import Iterator

import numpy as np


def raw_data_generator(
    data: np.ndarray,
    settings: dict,
    sfreq: int,
) -> Iterator[np.ndarray]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: int | float
    Returns
    -------
        np.array: new batch for run function of full segment length shape
    """
    sfreq_new = settings["sampling_rate_features_hz"]
    offset_time = settings["segment_length_features_ms"]
    offset_start = np.ceil(offset_time / 1000 * sfreq).astype(int)

    cnt_fsnew = 0
    for cnt in range(data.shape[1]+1):  # shape + 1 guarantees that the last sample is also included
        if cnt < offset_start:
            cnt_fsnew += 1
            continue

        cnt_fsnew += 1
        if cnt_fsnew >= (sfreq / sfreq_new):
            cnt_fsnew = 0
            yield data[:, cnt - offset_start : cnt]
