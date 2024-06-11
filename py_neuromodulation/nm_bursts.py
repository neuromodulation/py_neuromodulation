import numpy as np
from numpy.lib.function_base import _quantile as np_quantile  # type:ignore
from collections.abc import Sequence
from itertools import product

from pydantic import Field
from py_neuromodulation.nm_types import FeatureSelector, NMBaseModel
from py_neuromodulation.nm_features import NMFeature

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


LARGE_NUM = 2**24


def get_label_pos(burst_labels, valid_labels):
    max_label = np.max(burst_labels, axis=2).flatten()
    min_label = np.min(
        burst_labels, axis=2, initial=LARGE_NUM, where=burst_labels != 0
    ).flatten()
    label_positions = np.zeros_like(valid_labels)
    N = len(valid_labels)
    pos = 0
    i = 0
    while i < N:
        if valid_labels[i] >= min_label[pos] and valid_labels[i] <= max_label[pos]:
            label_positions[i] = pos
            i += 1
        else:
            pos += 1
    return label_positions


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
        self.segment_length_features_s = settings.segment_length_features_ms / 1000
        self.samples_overlap = int(
            self.sfreq
            * self.segment_length_features_s
            / settings.sampling_rate_features_hz
        )

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
        self.filter_data = self.bandpass_filter.filter_data

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

        self.need_duration_mean = (
            "duration" in self.used_features or "burst_rate_per_s" in self.used_features
        )

        self.batch = 0

        self.connectivity = np.zeros((3, 3, 3))
        self.connectivity[1, 1, :] = 1

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import hilbert
        from scipy.ndimage import label, sum_labels as label_sum, mean as label_mean

        filtered_data = np.abs(np.array(hilbert(self.filter_data(data))))

        # Update buffer array
        batch_size = (
            filtered_data.shape[-1] if self.batch == 0 else self.samples_overlap
        )
        self.batch += 1
        self.data_buffer = np.concatenate(
            (
                self.data_buffer,
                filtered_data[:, :, -batch_size:],
            ),
            axis=2,
        )[:, :, -self.num_max_samples_ring_buffer :]

        # Burst threshold is calculated with the percentile defined in the settings
        # Call low-level numpy function directly, extra checks not needed
        burst_thr = np_quantile(self.data_buffer, self.settings.threshold / 100)[
            :, :, None
        ]

        bursts = filtered_data >= burst_thr

        # Prepend False at the beginning ensures that data never starts on a burst
        # Floor division to ignore last burst if series ends in a burst (true burst length unknown)
        num_bursts = (
            np.sum(np.diff(bursts, axis=2, prepend=False), axis=2) // 2
        ).astype(np.float64)  # np.astype added to avoid casting error in np.divide

        # Label each burst with a unique id, limiting connectivity to last axis
        burst_labels = label(bursts, self.connectivity)[0]  # type: ignore

        # Remove labels of bursts that are at the end of the dataset, and 0
        labels_at_end = np.concatenate((np.unique(burst_labels[:, :, -1]), (0,)))
        valid_labels = np.unique(burst_labels)
        valid_labels = valid_labels[
            ~np.isin(valid_labels, labels_at_end, assume_unique=True)
        ]

        # Precalculate (channel, band) coordinates for each valid label
        label_positions = get_label_pos(burst_labels, valid_labels)

        # Now we're ready to calculate features
        burst_lengths = label_sum(bursts, burst_labels, index=valid_labels) / self.sfreq

        if "duration" in self.used_features:
            # Handle division by zero using np.divide
            self.burst_duration_mean = (
                np.divide(
                    np.sum(bursts, axis=2),
                    num_bursts,
                    out=np.zeros_like(num_bursts),
                    where=num_bursts != 0,
                )
                / self.sfreq
            )
            # The max needs to be calculated per channel
            # TODO: it might be interesting to write a C function for this
            duration_max_flat = np.zeros(self.n_channels * self.n_fbands)
            for idx in range(self.n_channels * self.n_fbands):
                duration_max_flat[idx] = np.max(
                    burst_lengths[label_positions == idx], initial=0
                )

            self.burst_duration_max = duration_max_flat.reshape(
                (self.n_channels, self.n_fbands)
            )

        if self.need_duration_mean:
            self.burst_amplitude_max = (filtered_data * bursts).max(axis=2)
            # The mean is actually a mean of means, so we need the mean for each individual burst
            label_means = label_mean(filtered_data, burst_labels, index=valid_labels)
            # TODO: it might be interesting to write a C function for this
            amplitude_mean_flat = np.zeros(self.n_channels * self.n_fbands)
            for idx in range(self.n_channels * self.n_fbands):
                mask = label_positions == idx
                amplitude_mean_flat[idx] = (
                    np.mean(label_means[mask]) if np.any(mask) else 0
                )

            self.burst_amplitude_mean = amplitude_mean_flat.reshape(
                (self.n_channels, self.n_fbands)
            )

        if "burst_rate_per_s" in self.used_features:
            self.burst_rate_per_s = (
                self.burst_duration_mean / self.segment_length_features_s
            )

        if "in_burst" in self.used_features:
            self.end_in_burst = bursts[:, :, -1]  # End in burst

        # Create dictionary to return
        for (ch_i, ch), (fb_i, fb), feat in self.feature_combinations:
            self.STORE_FEAT_DICT[feat](features_compute, ch_i, ch, fb_i, fb)

        return features_compute

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

    def store_in_burst(
        self, features_compute: dict, ch_i: int, ch: str, fb_i: int, fb: str
    ):
        features_compute[f"{ch}_bursts_{fb}_in_burst"] = self.end_in_burst[ch_i, fb_i]
