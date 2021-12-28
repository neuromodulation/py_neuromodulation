from typing import Optional

import numpy as np


def normalize_raw(raw_arr, normalize_samples, fs, method="mean", clip=False):
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
        n_idx = np.arange(0, raw_arr.shape[1], 1)
    else:
        n_idx = np.arange(
            raw_arr.shape[1] - normalize_samples, raw_arr.shape[1], 1
        )

    if method == "mean":
        mean_ = np.mean(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / mean_.T
    elif method == "median":
        median_ = np.median(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - median_) / median_.T
    elif method == "zscore":
        mean_ = np.mean(raw_arr[:, n_idx], axis=1)
        std_ = np.std(raw_arr[:, n_idx], axis=1)
        raw_norm = (raw_arr[:, -fs:].T - mean_) / std_.T
    else:
        raise ValueError(
            "Only `median`, `mean` and `zscore` are supported as "
            f"raw normalization methods. Got {method}."
        )

    if clip:
        if isinstance(clip, bool):
            clip = 3.0
        else:
            float(clip)
        raw_norm = raw_norm.clip(min=-clip, max=clip, dtype=float)

    return raw_norm.T


def normalize_features(
    current: np.ndarray,
    previous: Optional[np.ndarray],
    normalize_samples: int,
    method: str = "mean",
    clip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize features with respect to the past number of normalize_samples.

    Parameters
    ----------
    current : numpy array
        current features to normalize.
    previous : numpy array or None
        previous features, not normalized. Used for normalization of current features.
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
    current : numpy array
        normalized current features
    previous : numpy array
        previous features, not normalized.

    Raises
    ------
    ValueError
        returned  if method is not 'mean', 'median' or 'zscore'
    """
    if previous is None:
        return np.zeros_like(current), current
    sample_count = len(previous)
    idx = max(0, sample_count - normalize_samples)
    previous = np.vstack((previous, current))[idx:]

    if method == "mean":
        current = (current - previous.mean()) / previous.mean(ddof=0)
    elif method == "median":
        current = (current - previous.median()) / previous.median(ddof=0)
    elif method == "zscore":
        current = (current - np.mean(previous, axis=0)) / np.std(
            previous, axis=0
        )
    elif method == "zscore-median":
        current = (current - previous.median()) / previous.std(ddof=0)
    else:
        raise ValueError(
            "Only `median`, `mean`, `zscore` and `zscore-median` are supported as "
            f"feature normalization methods. Got {method}."
        )

    if clip:
        if isinstance(clip, bool):
            clip = 3.0
        else:
            float(clip)
        current = current.clip(min=-clip, max=clip)

    return current, previous
