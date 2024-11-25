"""Module for real-time data normalization."""

import numpy as np
from typing import TYPE_CHECKING, Callable, Literal, get_args

from py_neuromodulation.utils.pydantic_extensions import NMField
from py_neuromodulation.utils.types import (
    NMBaseModel,
    NORM_METHOD,
    NMPreprocessor,
)

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings

NormalizerType = Literal["raw", "feature"]


class NormalizationSettings(NMBaseModel):
    normalization_time_s: float = NMField(30, gt=0, custom_metadata={"unit": "s"})
    normalization_method: NORM_METHOD = NMField(default="zscore")
    clip: float = NMField(default=3, ge=0, custom_metadata={"unit": "a.u."})

    @staticmethod
    def list_normalization_methods() -> list[NORM_METHOD]:
        return list(get_args(NORM_METHOD))


class Normalizer(NMPreprocessor):
    def __init__(
        self,
        sfreq: float,
        settings: "NMSettings",
        type: NormalizerType,
    ) -> None:
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

        self.method = self.settings.normalization_method
        self.using_sklearn = self.method in ["quantile", "power", "robust", "minmax"]

        if self.using_sklearn:
            import sklearn.preprocessing as skpp

            NORM_METHODS_SKLEARN: dict[NORM_METHOD, Callable] = {
                "quantile": lambda: skpp.QuantileTransformer(n_quantiles=300),
                "robust": skpp.RobustScaler,
                "minmax": skpp.MinMaxScaler,
                "power": skpp.PowerTransformer,
            }

            self.normalizer = norm_sklearn(NORM_METHODS_SKLEARN[self.method]())

        else:
            NORM_FUNCTIONS = {
                "mean": norm_mean,
                "median": norm_median,
                "zscore": norm_zscore,
                "zscore-median": norm_zscore_median,
            }
            self.normalizer = NORM_FUNCTIONS[self.method]

    def process(self, data: np.ndarray) -> np.ndarray:
        # TODO: does feature normalization need to be transposed too?
        if self.type == "raw":
            data = data.T

        if self.previous.size == 0:  # Check if empty
            self.previous = data
            return data if self.type == "raw" else data.T

        self.previous = np.vstack((self.previous, data[-self.add_samples :]))

        data = self.normalizer(data, self.previous)

        if self.settings.clip:
            data = data.clip(min=-self.settings.clip, max=self.settings.clip)

        self.previous = self.previous[-self.num_samples_normalize + 1 :]

        data = np.nan_to_num(data)

        return data if self.type == "raw" else data.T


class RawNormalizer(Normalizer):
    def __init__(self, sfreq: float, settings: "NMSettings") -> None:
        super().__init__(sfreq, settings, "raw")


class FeatureNormalizer(Normalizer):
    def __init__(self, settings: "NMSettings") -> None:
        super().__init__(settings.sampling_rate_features_hz, settings, "feature")


""" Functions to check for NaN's before deciding which Numpy function to call """


def nan_mean(data: np.ndarray, axis: int) -> np.ndarray:
    return (
        np.nanmean(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.mean(data, axis=axis)
    )


def nan_std(data: np.ndarray, axis: int) -> np.ndarray:
    return (
        np.nanstd(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.std(data, axis=axis)
    )


def nan_median(data: np.ndarray, axis: int) -> np.ndarray:
    return (
        np.nanmedian(data, axis=axis)
        if np.any(np.isnan(sum(data)))
        else np.median(data, axis=axis)
    )


def norm_mean(current, previous):
    mean = nan_mean(previous, axis=0)
    return (current - mean) / mean


def norm_median(current, previous):
    median = nan_median(previous, axis=0)
    return (current - median) / median


def norm_zscore(current, previous):
    std = nan_std(previous, axis=0)
    std[std == 0] = 1  # same behavior as sklearn
    return (current - nan_mean(previous, axis=0)) / std


def norm_zscore_median(current, previous):
    std = nan_std(previous, axis=0)
    std[std == 0] = 1  # same behavior as sklearn
    return (current - nan_median(previous, axis=0)) / std


def norm_sklearn(sknormalizer):
    # For the following methods we check for the shape of current
    # when current is a 1D array, then it is the post-processing normalization,
    # and we need to expand, and remove the extra dimension afterwards
    # When current is a 2D array, then it is pre-processing normalization, and
    # there's no need for expanding.

    def sk_normalizer(current, previous):
        return (
            sknormalizer.fit(np.nan_to_num(previous))
            .transform(
                # if post-processing: pad dimensions to 2
                np.reshape(current, (2 - len(current.shape)) * (1,) + current.shape)
            )
            .squeeze()  # if post-processing: remove extra dimension # type: ignore
        )

    return sk_normalizer
