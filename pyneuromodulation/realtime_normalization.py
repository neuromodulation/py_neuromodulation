from numpy import arange, mean, median


def realtime_normalization(raw_arr, cnt_samples, normalize_samples, fs,
                           norm_type='mean'):
    """Normalization according to past normalize_samples according to mean or median.

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
    norm_type : str, optional
        data is normalization with subtract and divide by 'mean' or 'median' , by default 'mean'

    Returns
    -------
    raw_norm : ndarray
        normalized array

    Raises
    ------
    TypeError
        returned  if norm_type is not 'mean' or 'median'
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
    else: 
        raise TypeError("only median and mean are supported as normalization "
                        "method")
    raw_norm = (raw_arr[:, -fs:].T - norm_previous) / norm_previous.T

    return raw_norm.T
