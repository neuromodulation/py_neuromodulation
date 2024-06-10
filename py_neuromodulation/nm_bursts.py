import numpy as np
from collections.abc import Sequence
from itertools import product

from pydantic import Field
from py_neuromodulation.nm_types import FeatureSelector, NMBaseModel
from py_neuromodulation.nm_features import NMFeature

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


class BurstFeatures(FeatureSelector):
    duration: bool = True
    amplitude: bool = True
    burst_rate_per_s: bool = True
    in_burst: bool = True


class BurstSettings(NMBaseModel):
    threshold: float = Field(default=75, ge=0, le=100)
    time_duration_s: float = Field(default=30, ge=0)
    frequency_bands: list[str] = ["low beta", "high beta", "low gamma"]
    burst_features: BurstFeatures = BurstFeatures()


class Burst(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: float
    ) -> None:
        # Test settings
        for fband_burst in settings.burst_settings.frequency_bands:
            assert (
                fband_burst in list(settings.frequency_ranges_hz.keys())
            ), f"bursting {fband_burst} needs to be defined in settings['frequency_ranges_hz']"

        from py_neuromodulation.nm_filter import MNEFilter

        self.settings = settings.burst_settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.samples_overlap = int(
            self.sfreq
            * (settings.segment_length_features_ms / 1000)
            / settings.sampling_rate_features_hz
        )

        self.segment_length_features_s = settings.segment_length_features_ms / 1000

        self.fband_names = settings.burst_settings.frequency_bands
        f_ranges: list[tuple[float, float]] = [
            (
                settings.frequency_ranges_hz[fband_name][0],
                settings.frequency_ranges_hz[fband_name][1],
            )
            for fband_name in self.fband_names
        ]

        self.seglengths = np.floor(
            self.sfreq
            / 1000
            * np.array(
                [
                    settings.bandpass_filter_settings.segment_lengths_ms[fband]
                    for fband in self.fband_names
                ]
            )
        ).astype(int)

        self.num_max_samples_ring_buffer = int(
            self.sfreq * self.settings.time_duration_s
        )

        self.bandpass_filter = MNEFilter(
            f_ranges=f_ranges,
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        self.n_channels = len(self.ch_names)
        self.n_fbands = len(self.fband_names)

        # Create circular buffer array for previous time_duration_s
        self.data_buffer = np.empty(
            (self.n_channels, self.n_fbands, 0), dtype=np.float64
        )

        self.used_features = self.settings.burst_features.get_enabled()

        self.feature_combinations = list(
            product(
                enumerate(self.ch_names),
                enumerate(self.fband_names),
                self.settings.burst_features.get_enabled(),
            )
        )

        # Variables to store results
        self.burst_duration_mean: np.ndarray
        self.burst_duration_max: np.ndarray
        self.burst_amplitude_max: np.ndarray
        self.burst_amplitude_mean: np.ndarray
        self.burst_rate_per_s: np.ndarray
        self.end_in_burst: np.ndarray

        self.STORE_FEAT_DICT: dict[str, Callable] = {
            "duration": self.store_duration,
            "amplitude": self.store_amplitude,
            "burst_rate_per_s": self.store_burst_rate,
            "in_burst": self.store_in_burst,
        }

    def store_duration(
        self, features_compute: dict, ch_i: int, ch: str, fb_i: int, fb: str
    ):
        features_compute[f"{ch}_bursts_{fb}_duration_mean"] = self.burst_duration_mean[
            ch_i, fb_i
        ]

        features_compute[f"{ch}_bursts_{fb}_duration_max"] = self.burst_duration_max[
            ch_i, fb_i
        ]

    def store_amplitude(
        self, features_compute: dict, ch_i: int, ch: str, fb_i: int, fb: str
    ):
        features_compute[f"{ch}_bursts_{fb}_amplitude_mean"] = (
            self.burst_amplitude_mean[ch_i, fb_i]
        )
        features_compute[f"{ch}_bursts_{fb}_amplitude_max"] = self.burst_amplitude_max[
            ch_i, fb_i
        ]

    def store_burst_rate(
        self, features_compute: dict, ch_i: int, ch: str, fb_i: int, fb: str
    ):
        features_compute[f"{ch}_bursts_{fb}_burst_rate_per_s"] = self.burst_rate_per_s[
            ch_i, fb_i
        ]

    def store_in_burst(self, features_compute: dict, ch_i: int, ch: str, fb_i: int, fb: str):
        features_compute[f"{ch}_bursts_{fb}_in_burst"] = self.end_in_burst[ch_i, fb_i]

    @staticmethod
    def max_burst_duration(burst_labels: np.ndarray, burst_lengths: np.ndarray) -> int:
        return np.max([burst_lengths[label-1] for label in np.unique(burst_labels)])

    @staticmethod
    def mean_burst_amplitude(
        burst_labels: np.ndarray, burst_amplitude_mean: np.ndarray
    ) -> np.floating:
        return np.mean(
            [burst_amplitude_mean[label-1] for label in np.unique(burst_labels)]
        )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import hilbert
        from scipy.ndimage import label, sum_labels as label_sum, mean as label_mean

        filtered_data: np.ndarray = np.abs(
            np.array(hilbert(self.bandpass_filter.filter_data(data), axis=2))
        )

        n_channels, n_fbands, n_samples = filtered_data.shape

        # Update buffer array
        excess = max(
            0, self.data_buffer.shape[2] + n_samples - self.num_max_samples_ring_buffer
        )
        self.data_buffer = np.concatenate(
            (self.data_buffer[:, :, excess:], filtered_data), axis=2
        )

        # Detect bursts as values above threshold
        burst_thr = np.percentile(
            self.data_buffer, q=self.settings.threshold, axis=2, keepdims=True
        )
        bursts: np.ndarray = filtered_data >= burst_thr

        # Label each burst with a unique id: add a zero between each data series, flatten
        burst_labels, n_labels = label(
            np.concatenate(
                (bursts, np.zeros(bursts.shape[:2] + (1,), dtype=bool)), axis=2
            ).flatten()
        )  # type:ignore

        # Go back to original shape and remove zeros
        burst_labels = burst_labels.reshape(n_channels, n_fbands, n_samples + 1)[
            :, :, :-1
        ]

        # Get length of each burst, so we can get the max burst duration
        burst_lengths = label_sum(bursts, burst_labels, index=range(1, n_labels + 1))

        # Detect burst ends as places where the cumulative sum stops changing
        bursts_cumsum = np.cumsum(bursts, axis=2)
        burst_ends = np.concatenate(
            (
                np.zeros((n_channels, n_fbands, 2), dtype=bool),
                np.diff(bursts_cumsum, n=2) < 0,
            ),
            axis=2,
        )

        # num_bursts = np.sum(np.diff(bursts, axis = 2), axis = 2) // 2 # This is better is burst_ends not needed
        num_bursts = np.sum(burst_ends, axis=2)

        # Calculate duration features
        if "duration" in self.used_features:
            self.burst_duration_mean = np.sum(bursts, axis=2) / num_bursts
            self.burst_duration_max = np.apply_along_axis(
                self.max_burst_duration, 2, burst_labels, burst_lengths
            )

        # Calculate amplitude features
        if "amplitude" in self.used_features:
            self.burst_amplitude_max = (filtered_data * bursts).max(axis=2)
            # The mean is actually a mean of means, so we need the mean for each individual burst
            label_means = label_mean(
                filtered_data, burst_labels, index=range(1, n_labels + 1)
            )
            self.burst_amplitude_mean = np.apply_along_axis(
                self.mean_burst_amplitude, 2, burst_labels, label_means
            )

        # Burst rate per second
        if "burst_rate_per_s" in self.used_features:
            self.burst_rate_per_s = (
                self.burst_duration_mean / self.segment_length_features_s
            )

        # In burst
        if "in_burst" in self.used_features:
            self.end_in_burst = bursts[:, :, -1]

        # Create dictionary to return
        for (ch_i, ch), (fb_i, fb), feat in self.feature_combinations:
            self.STORE_FEAT_DICT[feat](features_compute, ch_i, ch, fb_i, fb)

        return features_compute

    @staticmethod
    def get_burst_amplitude_length(beta_averp_norm, burst_thr: float, sfreq: float):
        """
        Analysing the duration of beta burst
        """
        bursts = np.zeros((beta_averp_norm.shape[0] + 1), dtype=bool)
        bursts[1:] = beta_averp_norm >= burst_thr
        deriv = np.diff(bursts)
        burst_length = []
        burst_amplitude = []

        burst_time_points = np.where(deriv)[0]

        for i in range(burst_time_points.size // 2):
            burst_length.append(burst_time_points[2 * i + 1] - burst_time_points[2 * i])
            burst_amplitude.append(
                beta_averp_norm[burst_time_points[2 * i] : burst_time_points[2 * i + 1]]
            )

        # the last burst length (in case isburst == True) is omitted,
        # since the true burst length cannot be estimated
        return burst_amplitude, np.array(burst_length) / sfreq
