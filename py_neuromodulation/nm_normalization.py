"""Module for real-time data normalization."""
from enum import Enum

from sklearn import preprocessing
import numpy as np

from py_neuromodulation.nm_preprocessing import NMPreprocessor


class NORM_METHODS(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    ZSCORE = "zscore"
    ZSCORE_MEDIAN = "zscore-median"
    QUANTILE = "quantile"
    POWER = "power"
    ROBUST = "robust"
    MINMAX = "minmax"


def test_normalization_settings(
    normalization_time_s: int | float, normalization_method: str, clip: bool
):
    assert isinstance(
        normalization_time_s,
        (float, int),
    )

    assert isinstance(
        normalization_method, str
    ), "normalization method needs to be of type string"

    assert normalization_method in [e.value for e in NORM_METHODS], (
        f"select a valid normalization method, got {normalization_method}, "
        f"valid options are {[e.value for e in NORM_METHODS]}"
    )

    assert isinstance(clip, (float, int, bool))


class RawNormalizer(NMPreprocessor):
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

        test_normalization_settings(normalization_time_s, normalization_method, clip)

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

        test_normalization_settings(normalization_time_s, normalization_method, clip)

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

"""
Functions to check for NaN's before deciding which Numpy function to call
"""
def nan_mean(data, axis):
    return np.nanmean(data, axis=axis) if np.any(np.isnan(sum(data))) else np.mean(data, axis=axis)

def nan_std(data, axis):
    return np.nanstd(data, axis=axis) if np.any(np.isnan(sum(data))) else np.std(data, axis=axis)

def nan_median(data, axis):
    return np.nanmedian(data, axis=axis) if np.any(np.isnan(sum(data))) else np.median(data, axis=axis)

def _normalize_and_clip(
    current: np.ndarray,
    previous: np.ndarray,
    method: str,
    clip: int | float | bool,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data."""
    match method:
        case NORM_METHODS.MEAN.value:
            mean = nan_mean(previous, axis=0)
            current = (current - mean) / mean
        case NORM_METHODS.MEDIAN.value:
            median = nan_median(previous, axis=0)
            current = (current - median) / median
        case NORM_METHODS.ZSCORE.value:
            current = (current - nan_mean(previous, axis=0)) / nan_std(previous, axis=0)
        case NORM_METHODS.ZSCORE_MEDIAN.value:
            current = (current - nan_median(previous, axis=0)) / nan_std(previous, axis=0)
        # For the following methods we check for the shape of current
        # when current is a 1D array, then it is the post-processing normalization,
        # and we need to expand, and remove the extra dimension afterwards
        # When current is a 2D array, then it is pre-processing normalization, and
        # there's no need for expanding.
        case (NORM_METHODS.QUANTILE.value | 
              NORM_METHODS.ROBUST.value | 
              NORM_METHODS.MINMAX.value | 
              NORM_METHODS.POWER.value):
            
            norm_methods = {
                NORM_METHODS.QUANTILE.value : lambda: preprocessing.QuantileTransformer(n_quantiles=300),
                NORM_METHODS.ROBUST.value : preprocessing.RobustScaler,
                NORM_METHODS.MINMAX.value : preprocessing.MinMaxScaler,
                NORM_METHODS.POWER.value : preprocessing.PowerTransformer
            }
                
            current = (
                norm_methods[method]()
                .fit(np.nan_to_num(previous))
                .transform(
                    # if post-processing: pad dimensions to 2
                    np.reshape(current, (2-len(current.shape))*(1,) + current.shape)
                    )
                .squeeze() # if post-processing: remove extra dimension
            )
            
        case _:
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
