import numpy as np
from collections.abc import Iterable
from itertools import product

from py_neuromodulation.nm_types import NMBaseModel
from pydantic import model_validator
from typing import TYPE_CHECKING, Any, Callable

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import FeatureSelector, FrequencyRange

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


class PeakDetectionSettings(NMBaseModel):
    estimate: bool = True
    distance_troughs_ms: float = 10
    distance_peaks_ms: float = 5


class SharpwaveFeatures(FeatureSelector):
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


class SharpwaveSettings(NMBaseModel):
    sharpwave_features: SharpwaveFeatures = SharpwaveFeatures()
    filter_ranges_hz: list[FrequencyRange] = [
        FrequencyRange(5, 80),
        FrequencyRange(5, 30),
    ]
    detect_troughs: PeakDetectionSettings = PeakDetectionSettings()
    detect_peaks: PeakDetectionSettings = PeakDetectionSettings()
    estimator: dict[str, list[str]] = {
        "mean": ["interval"],
        "median": [],
        "max": ["prominence", "sharpness"],
        "min": [],
        "var": [],
    }
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
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: float
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

        self.used_features = self.sw_settings.sharpwave_features.get_enabled()

        # initializing estimator functions, respecitive for all sharpwave features
        self.estimator_dict: dict[str, dict[str, Callable]] = {
            feat: {
                est: getattr(np, est)
                for est in self.sw_settings.estimator.keys()
                if feat in self.sw_settings.estimator[est]
            }
            for feat_list in self.sw_settings.estimator.values()
            for feat in feat_list
        }

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

    def calc_feature(
        self,
        data: np.ndarray,
        features_compute: dict,
    ) -> dict:
        """Given a new data batch, the peaks, troughs and sharpwave features
        are estimated. Importantly only new data is being analyzed here. In
        steps of 1/settings["sampling_rate_features] analyzed and returned.
        Pre-initialized filters are applied to each channel.

        Parameters
        ----------
        data (np.ndarray): 2d data array with shape [num_channels, samples]
        features_compute (dict): Features.py estimated features

        Returns
        -------
        features_compute (dict): set features for Features.py object
        """
        dict_ch_features: dict[str, dict[str, float]] = {}

        combinations = product(
            enumerate(self.ch_names), self.list_filter, [False, True]
        )
        estimator_key_map: dict[str, Callable] = {}
        from scipy.signal import fftconvolve

        for (ch_idx, ch_name), (filter_name, filter), detect_troughs in combinations:
            self.data_process_sw = (
                data[ch_idx, :]
                if filter_name == "no_filter"
                else fftconvolve(data[ch_idx, :], filter, mode="same")
            )

            key_name_pt = "Trough" if detect_troughs else "Peak"

            if (not detect_troughs and not self.sw_settings.detect_peaks.estimate) or (
                detect_troughs and not self.sw_settings.detect_troughs.estimate
            ):
                continue

            # the detect_troughs loop start with peaks, s.t. data does not
            # need to be flipped
            if detect_troughs:
                self.data_process_sw = -self.data_process_sw

            self.analyze_waveform()

            # for each feature take the respective fun.
            for feature_name in self.used_features:
                for estimator_name, estimator in self.estimator_dict[
                    feature_name
                ].items():
                    key_name = f"{ch_name}_Sharpwave_{estimator_name.title()}_{feature_name}_{filter_name}"

                    # zero check because no peaks can be also detected
                    dict_ch_features.setdefault(key_name, {})[key_name_pt] = (
                        estimator(getattr(self, feature_name))
                        if len(getattr(self, feature_name)) != 0
                        else 0
                    )
                    estimator_key_map[key_name] = estimator

        if self.sw_settings.apply_estimator_between_peaks_and_troughs:
            # apply between 'Trough' and 'Peak' the respective function again
            # save only the 'est_fun' (e.g. max) between them

            # the key_name stays, since the estimator function stays between peaks and troughs
            for key_name, estimator in estimator_key_map.items():
                features_compute[key_name] = estimator(
                    [
                        list(dict_ch_features[key_name].values())[0],
                        list(dict_ch_features[key_name].values())[1],
                    ]
                )
        else:
            # otherwise, save all
            # write all "flatted" key value pairs in features_
            for key, subdict in dict_ch_features.items():
                for key_sub, value_sub in subdict.items():
                    features_compute[key + "_analyze_" + key_sub] = value_sub

        return features_compute

    def analyze_waveform(self) -> None:
        """Given the scipy.signal.find_peaks trough/peak distance
        settings specified sharpwave features are estimated.
        """

        from scipy.signal import find_peaks

        peak_idx = find_peaks(
            self.data_process_sw,
            distance=self.sw_settings.detect_troughs.distance_peaks_ms,
        )[0]
        trough_idx = find_peaks(
            -self.data_process_sw,
            distance=self.sw_settings.detect_troughs.distance_troughs_ms,
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

        if self.need_peak_left:
            self.peak_left: np.ndarray = self.data_process_sw[peak_idx_left]

        if self.need_peak_right:
            self.peak_right: np.ndarray = self.data_process_sw[peak_idx_right]

        if self.need_trough:
            self.trough: np.ndarray = self.data_process_sw[trough_idx]

        if self.sw_settings.sharpwave_features.interval:
            self.interval = np.concatenate((np.zeros(1), np.diff(trough_idx))) * (
                1000 / self.sfreq
            )

        if self.sw_settings.sharpwave_features.sharpness:
            # sharpess is calculated on a +- 5 ms window
            # valid troughs need 5 ms of margin on both siddes
            troughs_valid = trough_idx[
                np.logical_and(
                    trough_idx - int(5 * (1000 / self.sfreq)) > 0,
                    trough_idx + int(5 * (1000 / self.sfreq))
                    < self.data_process_sw.shape[0],
                )
            ]

            self.sharpness = (
                (
                    self.data_process_sw[troughs_valid]
                    - self.data_process_sw[troughs_valid - int(5 * (1000 / self.sfreq))]
                )
                + (
                    self.data_process_sw[troughs_valid]
                    - self.data_process_sw[troughs_valid + int(5 * (1000 / self.sfreq))]
                )
            ) / 2

        if self.need_steepness:
            # steepness is calculated as the first derivative
            steepness: np.ndarray = np.concatenate(
                (np.zeros(1), np.diff(self.data_process_sw))
            )

            if self.need_rise_steepness:
                # left peak -> trough
                # + 1 due to python syntax, s.t. the last element is included
                self.rise_steepness = np.array(
                    [
                        np.max(np.abs(steepness[peak_idx_left[i] : trough_idx[i] + 1]))
                        for i in range(trough_idx.size)
                    ]
                )

            if self.need_decay_steepness:
                # trough -> right peak
                self.decay_steepness = np.array(
                    [
                        np.max(
                            np.abs(steepness[trough_idx[i] : peak_idx_right[i] + 1])
                        )
                        for i in range(trough_idx.size)
                    ]
                )

            if self.sw_settings.sharpwave_features.slope_ratio:
                self.slope_ratio = self.rise_steepness - self.decay_steepness

        if self.sw_settings.sharpwave_features.prominence:
            self.prominence = np.abs(
                (self.peak_right + self.peak_left) / 2 - self.trough
            )

        if self.sw_settings.sharpwave_features.decay_time:
            self.decay_time = (peak_idx_left - trough_idx) * (1000 / self.sfreq)  # ms

        if self.sw_settings.sharpwave_features.rise_time:
            self.rise_time = (peak_idx_right - trough_idx) * (1000 / self.sfreq)  # ms

        if self.sw_settings.sharpwave_features.width:
            self.width = peak_idx_right - peak_idx_left  # ms
