from typing import Optional, Union

import numpy as np


def normalize_raw(
    current: np.ndarray,
    previous: Optional[np.ndarray],
    normalize_samples: int,
    sample_add: int,
    method: str = "mean",
    clip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data with respect to the past number of `normalize_samples`.

    Parameters
    ----------
    current : numpy array
        current raw data to normalize.
    previous : numpy array or None
        previous raw data, not normalized. Used for normalization of current data.
    normalize_samples : int
        number of past samples considered for normalization
    sample_add : int
        number of samples to add to previous
    method : str | default is 'mean'
        data is normalized via subtraction of the 'mean' or 'median' and
        subsequent division by the 'mean' or 'median'. For z-scoring enter
        'zscore'.
    clip : int | float, optional
        value at which to clip after normalization

    Returns
    -------
    current_norm : numpy array
        normalized array
    previous : numpy array
        previous features, not normalized.

    Raises
    ------
    ValueError
        returned  if norm_type is not 'mean', 'median' or 'zscore'
    """
    current = current.T
    if previous is None:
        previous = current
    else:
        previous = np.vstack((previous, current[-sample_add:]))
        previous = _transform_previous(
            previous=previous, normalize_samples=normalize_samples
        )

    current, previous = _normalize_and_clip(
        current=current,
        previous=previous,
        method=method,
        clip=clip,
        description="feature",
    )

    return current.T, previous


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

    previous = np.vstack((previous, current))
    previous = _transform_previous(
        previous=previous, normalize_samples=normalize_samples
    )

    current, previous = _normalize_and_clip(
        current=current,
        previous=previous,
        method=method,
        clip=clip,
        description="feature",
    )

    return current, previous


def _transform_previous(
    previous: np.ndarray, normalize_samples: int
) -> np.ndarray:
    """Crop previous data to reduce memory usage given normalization sample count."""
    sample_count = len(previous)
    idx = max(0, sample_count - normalize_samples)
    return previous[idx:]


def _normalize_and_clip(
    current: np.ndarray,
    previous: np.ndarray,
    method: str,
    clip: Union[int, float, bool],
    description: str,
) -> np.ndarray:
    """Normalize data."""
    if method == "mean":
        mean = previous.mean(axis=0)
        current = (current - mean) / mean
    elif method == "median":
        median = np.median(previous, axis=0)
        current = (current - median) / median
    elif method == "zscore":
        mean = previous.mean(axis=0)
        current = (current - previous.mean(axis=0)) / previous.std(axis=0)
    elif method == "zscore-median":
        current = (current - np.median(previous, axis=0)) / previous.std(
            axis=0
        )
    else:
        raise ValueError(
            "Only `median`, `mean`, `zscore` and `zscore-median` are supported as "
            f"{description} normalization methods. Got {method}."
        )

    if clip:
        current = _clip(data=current, clip=clip)
    return current, previous


def _clip(data: np.ndarray, clip: Union[bool, int, float]) -> np.ndarray:
    """Clip data."""
    if clip is True:
        clip = 3.0  # default value
    else:
        clip = float(clip)
    return data.clip(min=-clip, max=clip)
