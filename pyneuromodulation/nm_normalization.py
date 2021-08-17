from numpy import arange, mean, median, std


def normalize_raw(raw_arr, normalize_samples, fs, method='mean', clip=False):
    """Normalize data with respect to the past number of `normalize_samples`.

    Parameters
    ----------
    raw_arr : ndarray
        input array, shape (channels, time)
    normalize_samples : int
        number of past samples considered for normalization
    fs : int
        sampling frequency
    method : str | default is 'mean'
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
    if raw_arr.shape[1] < normalize_samples:
        n_idx = arange(0, raw_arr.shape[1], 1)
    else:
        n_idx = arange(
            raw_arr.shape[1] - normalize_samples, raw_arr.shape[1], 1)

    if method == "mean":
        mean_ = mean(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / mean_.T
    elif method == "median":
        median_ = median(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - median_) / median_.T
    elif method == "zscore":
        mean_ = mean(raw_arr[:, n_idx], axis=1)
        std_ = std(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / std_.T
    else:
        raise ValueError("Only `median`, `mean` and `zscore` are supported as "
                         f"raw normalization methods. Got {method}.")

    if clip:
        if isinstance(clip, bool):
            clip = 3.0
        else:
            float(clip)
        raw_norm = raw_norm.clip(min=-clip, max=clip, dtype=float)

    return raw_norm.T


def normalize_features(
        curr_, prev_, normalize_samples, method='mean', clip=False):
    """Normalize features with respect to the past number of normalize_samples.

    Parameters
    ----------
    curr_ : pandas.Series
        series of current features to normalize.
    prev_ : pandas.DataFrame
        data frame of all previous features, not normalized. These are used for
        normalization of the current features.
    normalize_samples : int
        number of past samples considered for normalization
    method : str | default is 'mean'
        data is normalized via subtraction of the 'mean' or 'median' and
        subsequent division by the 'mean' or 'median'. For z-scoring enter
        'zscore'.
    clip : int | float, optional
        value at which to clip on the lower and upper end after normalization.
        Useful for artifact rejection and handling of outliers.

    Returns
    -------
    curr_ : pandas.Series
        series of normalized current features

    Raises
    ------
    ValueError
        returned  if method is not 'mean', 'median' or 'zscore'
    """
    cnt_samples = len(prev_)
    if cnt_samples < normalize_samples:
        n_idx = arange(0, cnt_samples, 1)
    else:
        n_idx = arange(cnt_samples - normalize_samples, cnt_samples, 1)

    if method == 'mean':
        curr_ = (curr_ - prev_.iloc[n_idx].mean()) \
                / prev_.iloc[n_idx].mean(ddof=0)
    elif method == "median":
        curr_ = (curr_ - prev_.iloc[n_idx].median()) \
                / prev_.iloc[n_idx].median(ddof=0)
    elif method == 'zscore':
        curr_ = (curr_ - prev_.iloc[n_idx].mean()) \
                / prev_.iloc[n_idx].std(ddof=0)
    else:
        raise ValueError("Only `median`, `mean` and `zscore` are supported as "
                         f"feature normalization methods. Got {method}.")

    if clip:
        if isinstance(clip, bool):
            clip = 3.0
        else:
            float(clip)
        curr_.clip(lower=-clip, upper=clip, inplace=True)

    return curr_
