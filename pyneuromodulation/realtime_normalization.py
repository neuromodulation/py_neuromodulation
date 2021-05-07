from numpy import arange, mean, median
from scipy.stats import zscore


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
        return raw_arr
    if cnt_samples < normalize_samples:
        n_idx = arange(0, cnt_samples, 1)
    else:
        n_idx = arange(cnt_samples - normalize_samples, cnt_samples+1, 1)

    if norm_type == "mean":
        norm_previous = mean(raw_arr[:, n_idx], axis=1)
    elif norm_type == "median":
        norm_previous = median(raw_arr[:, n_idx], axis=1)
    elif norm_type == "zscore":
        norm_previous = zscore(raw_arr[:, n_idx], axis=1)
    else:
        raise ValueError("Only `median`, `mean` and `zscore` are supported as "
                         f"normalization methods. Got {norm_type}.")

    raw_norm = (raw_arr[:, -fs:].T - norm_previous) / norm_previous.T

    return raw_norm.T
