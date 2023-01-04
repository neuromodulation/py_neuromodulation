"""Module for real-time data normalization."""
from enum import Enum

import sklearn.preprocessing
import numpy as np


class NORM_METHODS(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    ZSCORE = "zscore"
    ZSCORE_MEDIAN = "zscore-median"
    QUANTILE = "quantile"
    POWER = "power"
    ROBUST = "robust"
    MINMAX = "minmax"


class RawNormalizer:
    def __init__(
        self,
        sfreq: int | float,
        sampling_rate_features_hz: int,
        normalization_method: str = "zscore",
        normalization_time_s: int | float = 30,
        clip: bool | int | float = False,
    ) -> None:
        """Normalize raw data.

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
        """
        self.method = normalization_method
        self.clip = clip
        self.num_samples_normalize = int(normalization_time_s * sfreq)
        self.add_samples = int(sfreq / sampling_rate_features_hz)
        self.previous = None

    def process(self, data: np.ndarray) -> np.ndarray:
        data = data.T
        if self.previous is None:
            self.previous = data
            return data.T

        self.previous = np.vstack((self.previous, data[-self.add_samples :]))

        data, self.previous = _normalize_and_clip(
            current=data,
            previous=self.previous,
            method=self.method,
            clip=self.clip,
            description="raw",
        )
        if self.previous.shape[0] >= self.num_samples_normalize:
            self.previous = self.previous[1:]

        return data.T


class FeatureNormalizer:
    def __init__(
        self,
        sampling_rate_features_hz: int,
        normalization_method: str = "zscore",
        normalization_time_s: int | float = 30,
        clip: bool | int | float = False,
    ) -> None:
        """Normalize raw data.

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
        """
        self.method = normalization_method
        self.clip = clip
        self.num_samples_normalize = int(
            normalization_time_s * sampling_rate_features_hz
        )
        self.previous = None

    def process(self, data: np.ndarray) -> np.ndarray:
        if self.previous is None:
            self.previous = data
            return data

        self.previous = np.vstack((self.previous, data))

        data, self.previous = _normalize_and_clip(
            current=data,
            previous=self.previous,
            method=self.method,
            clip=self.clip,
            description="feature",
        )
        if self.previous.shape[0] >= self.num_samples_normalize:
            self.previous = self.previous[1:]

        return data


def _normalize_and_clip(
    current: np.ndarray,
    previous: np.ndarray,
    method: str,
    clip: int | float | bool,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data."""
    if method == NORM_METHODS.MEAN.value:
        mean = np.nanmean(previous, axis=0)
        current = (current - mean) / mean
    elif method == NORM_METHODS.MEDIAN.value:
        median = np.nanmedian(previous, axis=0)
        current = (current - median) / median
    elif method == NORM_METHODS.ZSCORE.value:
        mean = np.nanmean(previous, axis=0)
        current = (current - mean) / np.nanstd(previous, axis=0)
    elif method == NORM_METHODS.ZSCORE_MEDIAN.value:
        current = (current - np.nanmedian(previous, axis=0)) / np.nanstd(
            previous, axis=0
        )
    elif method == NORM_METHODS.QUANTILE.value:
        current = (
            sklearn.preprocessing.QuantileTransformer(n_quantiles=300)
            .fit(np.nan_to_num(previous))
            .transform(np.expand_dims(current, axis=0))[0, :]
        )
    elif method == NORM_METHODS.ROBUST.value:
        current = (
            sklearn.preprocessing.RobustScaler()
            .fit(np.nan_to_num(previous))
            .transform(np.expand_dims(current, axis=0))[0, :]
        )
    elif method == NORM_METHODS.MINMAX.value:
        current = (
            sklearn.preprocessing.MinMaxScaler()
            .fit(np.nan_to_num(previous))
            .transform(np.expand_dims(current, axis=0))[0, :]
        )
    elif method == NORM_METHODS.POWER.value:
        current = (
            sklearn.preprocessing.PowerTransformer()
            .fit(np.nan_to_num(previous))
            .transform(np.expand_dims(current, axis=0))[0, :]
        )
    else:
        raise ValueError(
            f"Only {[e.value for e in NORM_METHODS]} are supported as "
            f"{description} normalization methods. Got {method}."
        )

    if clip:
        current = _clip(data=current, clip=clip)
    return current, previous


def _clip(data: np.ndarray, clip: bool | int | float) -> np.ndarray:
    """Clip data."""
    if clip is True:
        clip = 3.0  # default value
    else:
        clip = float(clip)
    return np.nan_to_num(data).clip(min=-clip, max=clip)
