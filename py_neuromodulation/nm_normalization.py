"""Module for real-time data normalization."""

from enum import StrEnum
from pydantic.dataclasses import dataclass
from pydantic import Field

import numpy as np

from py_neuromodulation.nm_preprocessing import NMPreprocessor


class NormMethod(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    ZSCORE = "zscore"
    ZSCORE_MEDIAN = "zscore-median"
    QUANTILE = "quantile"
    POWER = "power"
    ROBUST = "robust"
    MINMAX = "minmax"


@dataclass
class NormalizationSettings:
    normalization_time_s: float = 30
    normalization_method: NormMethod = NormMethod.ZSCORE
    clip: float = Field(default=3, ge=0)


class RawNormalizer(NMPreprocessor):
    def __init__(
        self,
        sfreq: float,
        sampling_rate_features_hz: float,
        settings: NormalizationSettings = NormalizationSettings()
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
        clip : float, optional
            value at which to clip after normalization
        """
        self.settings = settings

        self.num_samples_normalize = int(settings.normalization_time_s * sfreq)
        self.add_samples = int(sfreq / sampling_rate_features_hz)
        self.previous: np.ndarray = np.array([])  # Default empty array

    def process(self, data: np.ndarray) -> np.ndarray:
        data = data.T
        if self.previous.size == 0:  # Check if empty
            self.previous = data
            return data.T

        self.previous = np.vstack((self.previous, data[-self.add_samples :]))

        data, self.previous = _normalize_and_clip(
            current=data,
            previous=self.previous,
            method=self.settings.normalization_method,
            clip=self.settings.clip,
        )
        if self.previous.shape[0] >= self.num_samples_normalize:
            self.previous = self.previous[1:]

        return data.T


class FeatureNormalizer:
    def __init__(
        self,
        sampling_rate_features_hz: float,
        settings: NormalizationSettings = NormalizationSettings()
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
        clip : float, optional
            value at which to clip after normalization
        """
        self.settings = settings

        self.num_samples_normalize = int(
            settings.normalization_time_s * sampling_rate_features_hz
        )
        self.previous: np.ndarray = np.array([])

    def process(self, data: np.ndarray) -> np.ndarray:
        if self.previous.size == 0:
            self.previous = data
            return data

        self.previous = np.vstack((self.previous, data))

        data, self.previous = _normalize_and_clip(
            current=data,
            previous=self.previous,
            method=self.settings.normalization_method,
            clip=self.settings.clip,
        )
        if self.previous.shape[0] >= self.num_samples_normalize:
            self.previous = self.previous[1:]

        return data


"""
Functions to check for NaN's before deciding which Numpy function to call
"""


def nan_mean(data, axis):
    return (
        np.nanmean(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.mean(data, axis=axis)
    )


def nan_std(data, axis):
    return (
        np.nanstd(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.std(data, axis=axis)
    )


def nan_median(data, axis):
    return (
        np.nanmedian(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.median(data, axis=axis)
    )


def _normalize_and_clip(
    current: np.ndarray,
    previous: np.ndarray,
    method: NormMethod,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data."""
    match method:
        case NormMethod.MEAN:
            mean = nan_mean(previous, axis=0)
            current = (current - mean) / mean
        case NormMethod.MEDIAN:
            median = nan_median(previous, axis=0)
            current = (current - median) / median
        case NormMethod.ZSCORE:
            current = (current - nan_mean(previous, axis=0)) / nan_std(previous, axis=0)
        case NormMethod.ZSCORE_MEDIAN:
            current = (current - nan_median(previous, axis=0)) / nan_std(
                previous, axis=0
            )
        # For the following methods we check for the shape of current
        # when current is a 1D array, then it is the post-processing normalization,
        # and we need to expand, and remove the extra dimension afterwards
        # When current is a 2D array, then it is pre-processing normalization, and
        # there's no need for expanding.
        case (
            NormMethod.QUANTILE
            | NormMethod.ROBUST
            | NormMethod.MINMAX
            | NormMethod.POWER
        ):
            from sklearn.preprocessing import (
                QuantileTransformer,
                RobustScaler,
                MinMaxScaler,
                PowerTransformer,
            )

            norm_methods = {
                NormMethod.QUANTILE: lambda: QuantileTransformer(n_quantiles=300),
                NormMethod.ROBUST: RobustScaler,
                NormMethod.MINMAX: MinMaxScaler,
                NormMethod.POWER: PowerTransformer,
            }

            current = (
                norm_methods[method]()
                .fit(np.nan_to_num(previous))
                .transform(
                    # if post-processing: pad dimensions to 2
                    np.reshape(current, (2 - len(current.shape)) * (1,) + current.shape)
                )
                .squeeze()  # if post-processing: remove extra dimension
            )

    if clip != 0:
        current = _clip(data=current, clip=clip)
    return current, previous


def _clip(data: np.ndarray, clip: float) -> np.ndarray:
    """Clip data."""
    return np.nan_to_num(data).clip(min=-clip, max=clip)
