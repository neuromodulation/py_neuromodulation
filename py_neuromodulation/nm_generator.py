from typing import Iterator

import numpy as np


def ieeg_raw_generator(
    ieeg_raw, settings: dict, fs: int
) -> Iterator[np.ndarray]:
    """
    This generator function mimics online data acquisition.
    The nm_channels.csv selected raw channels are iteratively sampled with fs.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
    Returns
    -------
        np.array: new batch for run function of full segment length shape
    """

    cnt_fsnew = 0
    offset_time = settings["segment_length_features_ms"]
    offset_start = np.ceil(offset_time / 1000 * fs).astype(int)
    fs_new = settings["sampling_rate_features_hz"]

    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew += 1
            continue

        cnt_fsnew += 1
        if cnt_fsnew >= (fs / fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[:, cnt - offset_start : cnt]
