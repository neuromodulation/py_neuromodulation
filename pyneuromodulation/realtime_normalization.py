from numpy import arange, mean, median, std
from numpy import clip as np_clip


def realtime_normalization(raw_arr, cnt_samples, normalize_samples, fs,
                           norm_type='mean', clip=False):
    """Normalize data with respect to the past `normalize_samples`.

    Parameters
    ----------
    raw_arr : ndarray
        input array, shape (channels, time)
    cnt_samples : int
        current sample index
    normalize_samples : int
        number of past samples considered for normalization
    fs : int
        sampling frequency
    norm_type : str | default is 'mean'
        data is normalized via subtraction of the 'mean' or 'median' and
        subsequent division by the 'mean' or 'median'. For z-scoring enter
        'zscore'.
    clip : int | float, optional
        value at which to clip after normalization

    Returns
    -------
    raw_norm : ndarray
        normalized array

    Raises
    ------
    ValueError
        returned  if norm_type is not 'mean', 'median' or 'zscore'
    """

    if cnt_samples == 0:
        n_idx = arange(0, raw_arr.shape[1], 1)
    elif cnt_samples < normalize_samples:
        n_idx = arange(0, cnt_samples, 1)
    else:
        n_idx = arange(cnt_samples - normalize_samples + 1, cnt_samples + 1, 1)

    if norm_type == "mean":
        mean_ = mean(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / mean_.T
    elif norm_type == "median":
        median_ = median(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - median_) / median_.T
    elif norm_type == "zscore":
        mean_ = mean(raw_arr[:, n_idx], axis=1)
        std_ = std(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / std_.T
    else:
        raise ValueError("Only `median`, `mean` and `zscore` are supported as "
                         f"normalization methods. Got {norm_type}.")

    if clip:
        raw_norm = np_clip(raw_norm, a_min=-clip, a_max=clip)
        pass

    return raw_norm.T
