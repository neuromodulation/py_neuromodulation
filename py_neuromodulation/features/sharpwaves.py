from collections.abc import Sequence
from collections import defaultdict
from itertools import product

from pydantic import model_validator
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if np.__version__ >= "2.0.0":
    from numpy._core._methods import _mean as np_mean  # type: ignore
else:
    from numpy.core._methods import _mean as np_mean

from py_neuromodulation.utils.types import (
    NMFeature,
    NMBaseModel,
    BoolSelector,
    FrequencyRange,
)

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings

# Using low-level numpy mean function for performance, could do the same for the other estimators
ESTIMATOR_DICT = {
    "mean": np_mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
    "var": np.var,
}


class PeakDetectionSettings(NMBaseModel):
    estimate: bool = True
    distance_troughs_ms: float = 10
    distance_peaks_ms: float = 5


class SharpwaveFeatures(BoolSelector):
    peak_left: bool = False
    peak_right: bool = False
    trough: bool = False
    width: bool = False
    prominence: bool = True
    interval: bool = True
    decay_time: bool = False
    rise_time: bool = False
    sharpness: bool = True
    rise_steepness: bool = False
    decay_steepness: bool = False
    slope_ratio: bool = False


class SharpwaveEstimators(NMBaseModel):
    mean: list[str] = ["interval"]
    median: list[str] = []
    max: list[str] = ["prominence", "sharpness"]
    min: list[str] = []
    var: list[str] = []

    def keys(self):
        return ["mean", "median", "max", "min", "var"]

    def values(self):
        return [self.mean, self.median, self.max, self.min, self.var]


class SharpwaveSettings(NMBaseModel):
    sharpwave_features: SharpwaveFeatures = SharpwaveFeatures()
    filter_ranges_hz: list[FrequencyRange] = [
        FrequencyRange(5, 80),
        FrequencyRange(5, 30),
    ]
    detect_troughs: PeakDetectionSettings = PeakDetectionSettings()
    detect_peaks: PeakDetectionSettings = PeakDetectionSettings()
    estimator: SharpwaveEstimators = SharpwaveEstimators()
    apply_estimator_between_peaks_and_troughs: bool = True

    def disable_all_features(self):
        self.sharpwave_features.disable_all()
        for est in self.estimator.keys():
            self.estimator[est] = []

    @model_validator(mode="after")
    def test_settings(cls, settings):
        # check if all features are also enabled via an estimator
        estimator_list = [est for list_ in settings.estimator.values() for est in list_]

        for used_feature in settings.sharpwave_features.get_enabled():
            assert (
                used_feature in estimator_list
            ), f"Add estimator key for {used_feature}"

        return settings


class SharpwaveAnalyzer(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: float
    ) -> None:
        self.sw_settings = settings.sharpwave_analysis_settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.list_filter: list[tuple[str, Any]] = []
        self.trough: list = []
        self.troughs_idx: list = []

        settings.validate()

        # FrequencyRange's are already ensured to have high > low
        # Test that the higher frequency is smaller than the sampling frequency
        for filter_range in settings.sharpwave_analysis_settings.filter_ranges_hz:
            assert filter_range[1] < sfreq, (
                "Filter range has to be smaller than sfreq, "
                f"got sfreq {sfreq} and filter range {filter_range}"
            )

        for filter_range in settings.sharpwave_analysis_settings.filter_ranges_hz:
            # Test settings
            # TODO: handle None values
            if filter_range[0] is None:
                self.list_filter.append(("no_filter", None))
            else:
                from mne.filter import create_filter

                self.list_filter.append(
                    (
                        f"range_{filter_range[0]:.0f}_{filter_range[1]:.0f}",
                        create_filter(
                            None,
                            sfreq,
                            l_freq=filter_range[0],
                            h_freq=filter_range[1],
                            fir_design="firwin",
                            # l_trans_bandwidth=None,
                            # h_trans_bandwidth=None,
                            # filter_length=str(sfreq) + "ms",
                            verbose=False,
                        ),
                    )
                )

        self.filter_names = [name for name, _ in self.list_filter]
        self.filters = np.vstack([filter for _, filter in self.list_filter])
        self.filters = np.tile(self.filters[None, :, :], (len(self.ch_names), 1, 1))

        self.used_features = self.sw_settings.sharpwave_features.get_enabled()

        # initializing estimator functions, respecitive for all sharpwave features
        self.estimator_dict: dict[str, dict[str, Callable]] = {
            feat: {
                est: ESTIMATOR_DICT[est]
                for est in self.sw_settings.estimator.keys()
                if feat in self.sw_settings.estimator[est]
            }
            for feat_list in self.sw_settings.estimator.values()
            for feat in feat_list
        }

        estimator_combinations = [
            (feature_name, estimator_name, estimator)
            for feature_name in self.used_features
            for estimator_name, estimator in self.estimator_dict[feature_name].items()
        ]

        filter_combinations = list(
            product(
                enumerate(self.ch_names), enumerate(self.filter_names), [False, True]
            )
        )

        self.estimator_key_map: dict[str, Callable] = {}
        self.combinations = []
        for (ch_idx, ch_name), (
            filter_idx,
            filter_name,
        ), detect_troughs in filter_combinations:
            for feature_name, estimator_name, estimator in estimator_combinations:
                key_name = f"{ch_name}_Sharpwave_{estimator_name.title()}_{feature_name}_{filter_name}"
                self.estimator_key_map[key_name] = estimator
            self.combinations.append(
                (
                    (ch_idx, ch_name),
                    (filter_idx, filter_name),
                    detect_troughs,
                    estimator_combinations,
                )
            )

        # Check required feature computations according to settings
        self.need_peak_left = (
            self.sw_settings.sharpwave_features.peak_left
            or self.sw_settings.sharpwave_features.prominence
        )
        self.need_peak_right = (
            self.sw_settings.sharpwave_features.peak_right
            or self.sw_settings.sharpwave_features.prominence
        )
        self.need_trough = (
            self.sw_settings.sharpwave_features.trough
            or self.sw_settings.sharpwave_features.prominence
        )

        self.need_decay_steepness = (
            self.sw_settings.sharpwave_features.decay_steepness
            or self.sw_settings.sharpwave_features.slope_ratio
        )

        self.need_rise_steepness = (
            self.sw_settings.sharpwave_features.rise_steepness
            or self.sw_settings.sharpwave_features.slope_ratio
        )

        self.need_steepness = self.need_rise_steepness or self.need_decay_steepness

    def calc_feature(self, data: np.ndarray) -> dict:
        """Given a new data batch, the peaks, troughs and sharpwave features
        are estimated. Importantly only new data is being analyzed here. In
        steps of 1/settings["sampling_rate_features] analyzed and returned.
        Pre-initialized filters are applied to each channel.

        Parameters
        ----------
        data (np.ndarray): 2d data array with shape [num_channels, samples]
        feature_results (dict): Features.py estimated features

        Returns
        -------
        feature_results (dict): set features for Features.py object
        """
        dict_ch_features: dict[str, dict[str, float]] = defaultdict(lambda: {})

        from scipy.signal import fftconvolve

        data = np.tile(data[:, None, :], (1, len(self.list_filter), 1))
        data = fftconvolve(data, self.filters, axes=2, mode="same")

        self.filtered_data = (
            data  # TONI: Expose filtered data for example 3, need a better way
        )

        feature_results = {}

        for (
            (ch_idx, ch_name),
            (filter_idx, filter_name),
            detect_troughs,
            estimator_combinations,
        ) in self.combinations:
            sub_data = data[ch_idx, filter_idx, :]

            key_name_pt = "Trough" if detect_troughs else "Peak"

            if (not detect_troughs and not self.sw_settings.detect_peaks.estimate) or (
                detect_troughs and not self.sw_settings.detect_troughs.estimate
            ):
                continue

            # the detect_troughs loop start with peaks, s.t. data does not need to be flipped
            sub_data = -sub_data if detect_troughs else sub_data
            # sub_data *= 1 - 2 * detect_troughs # branchless version

            waveform_results = self.analyze_waveform(sub_data)

            # for each feature take the respective fun.
            for feature_name, estimator_name, estimator in estimator_combinations:
                feature_data = waveform_results[feature_name]
                key_name = f"{ch_name}_Sharpwave_{estimator_name.title()}_{feature_name}_{filter_name}"

                # zero check because no peaks can be also detected
                feature_data = estimator(feature_data) if len(feature_data) != 0 else 0
                dict_ch_features[key_name][key_name_pt] = feature_data

        if self.sw_settings.apply_estimator_between_peaks_and_troughs:
            # apply between 'Trough' and 'Peak' the respective function again
            # save only the 'est_fun' (e.g. max) between them

            # the key_name stays, since the estimator function stays between peaks and troughs
            for key_name, estimator in self.estimator_key_map.items():
                feature_results[key_name] = estimator(
                    [
                        list(dict_ch_features[key_name].values())[0],
                        list(dict_ch_features[key_name].values())[1],
                    ]
                )
        else:
            # otherwise, save all write all "flattened" key value pairs in feature_results
            for key, subdict in dict_ch_features.items():
                for key_sub, value_sub in subdict.items():
                    feature_results[key + "_analyze_" + key_sub] = value_sub

        return feature_results

    def analyze_waveform(self, data) -> dict:
        """Given the scipy.signal.find_peaks trough/peak distance
        settings specified sharpwave features are estimated.
        """

        from scipy.signal import find_peaks

        # TODO: find peaks is actually not that big a performance hit, but the rest
        # of this function is. Perhaps find_peaks can be put in a loop and the rest optimized somehow?
        peak_idx: np.ndarray = find_peaks(
            data, distance=self.sw_settings.detect_troughs.distance_peaks_ms
        )[0]
        trough_idx: np.ndarray = find_peaks(
            -data, distance=self.sw_settings.detect_troughs.distance_troughs_ms
        )[0]

        """ Find left and right peak indexes for each trough """
        peak_pointer = first_valid = last_valid = 0
        peak_idx_left_list: list[int] = []
        peak_idx_right_list: list[int] = []

        for i in range(len(trough_idx)):
            # Locate peak right of current trough
            while (
                peak_pointer < peak_idx.size and peak_idx[peak_pointer] < trough_idx[i]
            ):
                peak_pointer += 1

            if peak_pointer - 1 < 0:
                # If trough has no peak to it's left, it's not valid
                first_valid = i + 1  # Try with next one
                continue

            if peak_pointer == peak_idx.size:
                # If we went past the end of the peaks list, trough had no peak to its right
                continue

            last_valid = i
            peak_idx_left_list.append(peak_idx[peak_pointer - 1])
            peak_idx_right_list.append(peak_idx[peak_pointer])

        # Remove non valid troughs and make array of left and right peaks for  each trough
        trough_idx = trough_idx[first_valid : last_valid + 1]
        peak_idx_left = np.array(peak_idx_left_list, dtype=int)
        peak_idx_right = np.array(peak_idx_right_list, dtype=int)

        """ Calculate features (vectorized) """
        results: dict = {}

        if self.need_peak_left:
            results["peak_left"] = data[peak_idx_left]

        if self.need_peak_right:
            results["peak_right"] = data[peak_idx_right]

        if self.need_trough:
            results["trough"] = data[trough_idx]

        if self.sw_settings.sharpwave_features.interval:
            results["interval"] = np.concatenate((np.zeros(1), np.diff(trough_idx))) * (
                1000 / self.sfreq
            )

        if self.sw_settings.sharpwave_features.sharpness:
            # sharpess is calculated on a +- 5 ms window
            # valid troughs need 5 ms of margin on both sides
            troughs_valid = trough_idx[
                np.logical_and(
                    trough_idx - int(5 * (1000 / self.sfreq)) > 0,
                    trough_idx + int(5 * (1000 / self.sfreq)) < data.shape[0],
                )
            ]
            trough_height = data[troughs_valid]
            left_height = data[troughs_valid - int(5 * (1000 / self.sfreq))]
            right_height = data[troughs_valid + int(5 * (1000 / self.sfreq))]
            # results["sharpness"] = ((trough_height - left_height) + (trough_height - right_height)) / 2
            results["sharpness"] = trough_height - 0.5 * (left_height + right_height)

        if self.need_steepness:
            # steepness is calculated as the first derivative
            steepness: np.ndarray = np.concatenate((np.zeros(1), np.diff(data)))

            # Create an array with the rise and decay steepness for each trough
            # 0th dimension for rise/decay, 1st for trough index, 2nd for timepoint
            steepness_troughs = np.zeros((2, trough_idx.shape[0], steepness.shape[0]))
            if self.need_rise_steepness or self.need_decay_steepness:
                for i in range(len(trough_idx)):
                    steepness_troughs[
                        0, i, 0 : trough_idx[i] - peak_idx_left[i] + 1
                    ] = steepness[peak_idx_left[i] : trough_idx[i] + 1]
                    steepness_troughs[
                        1, i, 0 : peak_idx_right[i] - trough_idx[i] + 1
                    ] = steepness[trough_idx[i] : peak_idx_right[i] + 1]

            if self.need_rise_steepness:
                # left peak -> trough
                # + 1 due to python syntax, s.t. the last element is included
                results["rise_steepness"] = np.max(
                    np.abs(steepness_troughs[0, :, :]), axis=1
                )

            if self.need_decay_steepness:
                # trough -> right peak
                results["decay_steepness"] = np.max(
                    np.abs(steepness_troughs[1, :, :]), axis=1
                )

            if self.sw_settings.sharpwave_features.slope_ratio:
                results["slope_ratio"] = (
                    results["rise_steepness"] - results["decay_steepness"]
                )

        if self.sw_settings.sharpwave_features.prominence:
            results["prominence"] = np.abs(
                (results["peak_right"] + results["peak_left"]) / 2 - results["trough"]
            )

        if self.sw_settings.sharpwave_features.decay_time:
            results["decay_time"] = (peak_idx_left - trough_idx) * (
                1000 / self.sfreq
            )  # ms

        if self.sw_settings.sharpwave_features.rise_time:
            results["rise_time"] = (peak_idx_right - trough_idx) * (
                1000 / self.sfreq
            )  # ms

        if self.sw_settings.sharpwave_features.width:
            results["width"] = peak_idx_right - peak_idx_left  # ms

        return results
