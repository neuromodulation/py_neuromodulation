"""Module for real-time data normalization."""

import numpy as np
from typing import TYPE_CHECKING, Callable, Literal, get_args

from py_neuromodulation.nm_types import NMBaseModel, Field, NormMethod
from py_neuromodulation.nm_preprocessing import NMPreprocessor


if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings

NormalizerType = Literal["raw", "feature"]


class NormalizationSettings(NMBaseModel):
    normalization_time_s: float = 30
    normalization_method: NormMethod = "zscore"
    clip: float = Field(default=3, ge=0)

    @staticmethod
    def list_normalization_methods() -> list[NormMethod]:
        return list(get_args(NormMethod))


class Normalizer(NMPreprocessor):
    def __init__(
        self,
        sfreq: float,
        settings: "NMSettings",
        type: NormalizerType,
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

        self.type = type
        self.settings: NormalizationSettings

        match self.type:
            case "raw":
                self.settings = settings.raw_normalization_settings.validate()
                self.add_samples = int(sfreq / settings.sampling_rate_features_hz)
            case "feature":
                self.settings = settings.feature_normalization_settings.validate()
                self.add_samples = 0

        # For type = "feature" sfreq = sampling_rate_features_hz
        self.num_samples_normalize = int(self.settings.normalization_time_s * sfreq)

        self.previous: np.ndarray = np.empty((0, 0))  # Default empty array

    def process(self, data: np.ndarray) -> np.ndarray:
        # TODO: does feature normalization need to be transposed too?
        if self.type == "raw":
            data = data.T

        if self.previous.size == 0:  # Check if empty
            self.previous = data
            return data if self.type == "raw" else data.T

        self.previous = np.vstack((self.previous, data[-self.add_samples :]))

        data, self.previous = _normalize_and_clip(
            current=data,
            previous=self.previous,
            method=self.settings.normalization_method,
            clip=self.settings.clip,
        )

        if self.previous.shape[0] >= self.num_samples_normalize:
            self.previous = self.previous[1:]

        return data if self.type == "raw" else data.T


class RawNormalizer(Normalizer):
    def __init__(self, sfreq: float, settings: "NMSettings") -> None:
        super().__init__(sfreq, settings, "raw")


class FeatureNormalizer(Normalizer):
    def __init__(self, settings: "NMSettings") -> None:
        super().__init__(settings.sampling_rate_features_hz, settings, "feature")


""" Functions to check for NaN's before deciding which Numpy function to call """


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
        case "mean":
            mean = nan_mean(previous, axis=0)
            current = (current - mean) / mean
        case "median":
            median = nan_median(previous, axis=0)
            current = (current - median) / median
        case "zscore":
            current = (current - nan_mean(previous, axis=0)) / nan_std(previous, axis=0)
        case "zscore-median":
            current = (current - nan_median(previous, axis=0)) / nan_std(
                previous, axis=0
            )
        # For the following methods we check for the shape of current
        # when current is a 1D array, then it is the post-processing normalization,
        # and we need to expand, and remove the extra dimension afterwards
        # When current is a 2D array, then it is pre-processing normalization, and
        # there's no need for expanding.
        case "quantile" | "power" | "robust" | "minmax":
            from sklearn.preprocessing import (
                QuantileTransformer,
                RobustScaler,
                MinMaxScaler,
                PowerTransformer,
            )

            norm_methods: dict[NormMethod, Callable] = {
                "quantile": lambda: QuantileTransformer(n_quantiles=300),
                "robust": RobustScaler,
                "minmax": MinMaxScaler,
                "power": PowerTransformer,
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

    if clip:
        current = np.nan_to_num(current).clip(min=-clip, max=clip)

    return current, previous
