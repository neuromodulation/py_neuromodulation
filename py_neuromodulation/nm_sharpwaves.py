import numpy as np
from scipy import signal
from mne.filter import create_filter
from typing import Iterable
import sys

from py_neuromodulation import nm_features_abc


class NoValidTroughException(Exception):
    pass


class SharpwaveAnalyzer(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:

        self.sw_settings = settings["sharpwave_analysis_settings"]
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.list_filter = []
        for filter_range in settings["sharpwave_analysis_settings"][
            "filter_ranges_hz"
        ]:

            if filter_range[0] is None:
                self.list_filter.append(("no_filter", None))
            else:
                self.list_filter.append(
                    (
                        f"range_{filter_range[0]}_{filter_range[1]}",
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

        # initialize used features
        self.used_features = list()
        for feature_name, val in self.sw_settings["sharpwave_features"].items():
            if val is True:
                self.used_features.append(feature_name)

        # initialize attributes
        self._initialize_sw_features()

        # initializing estimator functions, respecitive for all sharpwave features
        fun_names = []
        for used_feature in self.used_features:
            estimator_list_feature = (
                []
            )  # one feature can have multiple estimators
            for estimator, est_features in self.sw_settings[
                "estimator"
            ].items():
                if est_features is not None:
                    for est_feature in est_features:
                        if used_feature == est_feature:
                            estimator_list_feature.append(estimator)
            fun_names.append(estimator_list_feature)

        self.estimator_names = fun_names
        self.estimator_functions = [
            [
                getattr(np, est_name)
                for est_name in self.estimator_names[feature_idx]
            ]
            for feature_idx in range(len(self.estimator_names))
        ]

    def _initialize_sw_features(self) -> None:
        """Resets used attributes to empty lists"""
        for feature_name in self.used_features:
            setattr(self, feature_name, list())
        if "trough" not in self.used_features:
            # trough attribute is still necessary, even if it is not specified in settings
            self.trough = list()
        self.troughs_idx = list()

    def _get_peaks_around(self, trough_ind, arr_ind_peaks, filtered_dat):
        """Find the closest peaks to the right and left side a given trough.

        Parameters
        ----------
        trough_ind (int): index of trough
        arr_ind_peaks (np.ndarray): array of peak indices
        filtered_dat (np.ndarray): raw data batch

        Raises:
            NoValidTroughException: Returned if no adjacent peak can be found

        Returns
        -------
        peak_left_idx (np.ndarray): index of left peak
        peak_right_idx (np.ndarray): index of right peak
        peak_left_val (np.ndarray): value of left peak
        peak_right_val (np.ndarray): value of righ peak
        """

        try: peak_right_idx = arr_ind_peaks[arr_ind_peaks > trough_ind][0]
        except IndexError: raise NoValidTroughException("No valid trough")

        try: peak_left_idx  = arr_ind_peaks[arr_ind_peaks < trough_ind][-1]
        except IndexError: raise NoValidTroughException("No valid trough")
       
        return (
            peak_left_idx,
            peak_right_idx,
            filtered_dat[peak_left_idx],
            filtered_dat[peak_right_idx],
        )

    def calc_feature(
        self,
        data: np.array,
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
        for ch_idx, ch_name in enumerate(self.ch_names):
            for filter_name, filter in self.list_filter:
                if filter_name == "no_filter":
                    self.data_process_sw = data[ch_idx, :]
                else:
                    self.data_process_sw = signal.convolve(
                        data[ch_idx, :], filter, mode="same"
                    )

                # check settings if troughs and peaks are analyzed

                dict_ch_features = {}

                for detect_troughs in [False, True]:

                    if detect_troughs is False:
                        if (
                            self.sw_settings["detect_peaks"]["estimate"]
                            is False
                        ):
                            continue
                        key_name_pt = "Peak"
                        # the detect_troughs loop start with peaks, s.t. data does not
                        # need to be flipped

                    if detect_troughs is True:
                        if (
                            self.sw_settings["detect_troughs"]["estimate"]
                            is False
                        ):
                            continue
                        key_name_pt = "Trough"

                        self.data_process_sw = -self.data_process_sw

                    self._initialize_sw_features()  # reset sharpwave feature attriubtes to empty lists
                    self.analyze_waveform()

                    # for each feature take the respective fun.
                    for feature_idx, feature_name in enumerate(
                        self.used_features
                    ):
                        for est_idx, estimator_name in enumerate(
                            self.estimator_names[feature_idx]
                        ):
                            key_name = "_".join(
                                [
                                    ch_name,
                                    "Sharpwave",
                                    self.estimator_names[feature_idx][
                                        est_idx
                                    ].title(),
                                    feature_name,
                                    filter_name,
                                ]
                            )
                            # zero check because no peaks can be also detected
                            val = (
                                self.estimator_functions[feature_idx][est_idx](
                                    getattr(self, feature_name)
                                )
                                if len(getattr(self, feature_name)) != 0
                                else 0
                            )
                            if key_name not in dict_ch_features:
                                dict_ch_features[key_name] = {}
                            dict_ch_features[key_name][key_name_pt] = val

                if self.sw_settings[
                    "apply_estimator_between_peaks_and_troughs"
                ]:
                    # apply between 'Trough' and 'Peak' the respective function again
                    # save only the 'est_fun' (e.g. max) between them

                    for idx, key_name in enumerate(dict_ch_features):
                        # the key_name stays, since the estimator function stays between peaks and troughs
                        # this array needs to be flattened
                        features_compute[key_name] = np.concatenate(
                            self.estimator_functions
                        )[idx](
                            [
                                list(dict_ch_features[key_name].values())[0],
                                list(dict_ch_features[key_name].values())[1],
                            ]
                        )

                else:
                    # otherwise, save all
                    # write all "flatted" key value pairs in features_
                    for key, value in dict_ch_features.items():
                        for key_sub, value_sub in dict_ch_features[key].items():
                            features_compute[
                                key + "_analyze_" + key_sub
                            ] = value_sub

        return features_compute

    def analyze_waveform(self) -> None:
        """Given the scipy.signal.find_peaks  trough/peak distance
        settings specified sharpwave features are estimated.

        Parameters
        ----------

        Raises:
            NoValidTroughException: Return if no adjacent peak can be found
            NoValidTroughException: Return if no adjacent peak can be found

        """

        peaks = signal.find_peaks(
            self.data_process_sw,
            distance=self.sw_settings["detect_troughs"]["distance_peaks_ms"],
        )[0]
        troughs = signal.find_peaks(
            -self.data_process_sw,
            distance=self.sw_settings["detect_troughs"]["distance_troughs_ms"],
        )[0]

        right_peak_idx = 0
        for trough_idx in troughs:
            
            while right_peak_idx < peaks.size and peaks[right_peak_idx] < trough_idx:
                right_peak_idx += 1

            if right_peak_idx - 1 < 0:
                continue
            peak_idx_left = peaks[right_peak_idx - 1]

            if right_peak_idx >= peaks.size:
                continue
            peak_idx_right = peaks[right_peak_idx]

            peak_left = self.data_process_sw[peak_idx_left]
            peak_right = self.data_process_sw[peak_idx_right]

            trough = self.data_process_sw[trough_idx]
            self.trough.append(trough)
            self.troughs_idx.append(trough_idx)

            if self.sw_settings["sharpwave_features"]["interval"] is True:
                if len(self.troughs_idx) > 1:
                    # take the last identified trough idx
                    # corresponds here to second last trough_idx

                    interval = (trough_idx - self.troughs_idx[-2]) * (
                        1000 / self.sfreq
                    )
                else:
                    # set first interval to zero
                    interval = 0
                self.interval.append(interval)

            if self.sw_settings["sharpwave_features"]["peak_left"] is True:
                self.peak_left.append(peak_left)

            if self.sw_settings["sharpwave_features"]["peak_right"] is True:
                self.peak_right.append(peak_right)

            if self.sw_settings["sharpwave_features"]["sharpness"] is True:
                # check if sharpness can be calculated
                # trough_idx 5 ms need to be consistent
                if (trough_idx - int(5 * (1000 / self.sfreq)) <= 0) or (
                    trough_idx + int(5 * (1000 / self.sfreq))
                    >= self.data_process_sw.shape[0]
                ):
                    continue

                sharpness = (
                    (
                        self.data_process_sw[trough_idx]
                        - self.data_process_sw[
                            trough_idx - int(5 * (1000 / self.sfreq))
                        ]
                    )
                    + (
                        self.data_process_sw[trough_idx]
                        - self.data_process_sw[
                            trough_idx + int(5 * (1000 / self.sfreq))
                        ]
                    )
                ) / 2

                self.sharpness.append(sharpness)

            if self.sw_settings["sharpwave_features"]["rise_steepness"] is True:
                # steepness is calculated as the first derivative
                # from peak/trough to trough/peak
                # here  + 1 due to python syntax, s.t. the last element is included
                rise_steepness = np.max(
                    np.diff(
                        self.data_process_sw[peak_idx_left : trough_idx + 1]
                    )
                )
                self.rise_steepness.append(rise_steepness)

            if (
                self.sw_settings["sharpwave_features"]["decay_steepness"]
                is True
            ):
                decay_steepness = np.max(
                    np.diff(
                        self.data_process_sw[trough_idx : peak_idx_right + 1]
                    )
                )
                self.decay_steepness.append(decay_steepness)

            if (
                self.sw_settings["sharpwave_features"]["rise_steepness"] is True
                and self.sw_settings["sharpwave_features"]["decay_steepness"]
                is True
                and self.sw_settings["sharpwave_features"]["slope_ratio"]
                is True
            ):
                self.slope_ratio.append(rise_steepness - decay_steepness)

            if self.sw_settings["sharpwave_features"]["prominence"] is True:
                self.prominence.append(
                    np.abs(
                        (peak_right + peak_left) / 2
                        - self.data_process_sw[trough_idx]
                    )
                )

            if self.sw_settings["sharpwave_features"]["decay_time"] is True:
                self.decay_time.append(
                    (peak_idx_left - trough_idx) * (1000 / self.sfreq)
                )  # ms

            if self.sw_settings["sharpwave_features"]["rise_time"] is True:
                self.rise_time.append(
                    (peak_idx_right - trough_idx) * (1000 / self.sfreq)
                )  # ms

            if self.sw_settings["sharpwave_features"]["width"] is True:
                self.width.append(peak_idx_right - peak_idx_left)  # ms

    @staticmethod
    def test_settings(
        s: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        for filter_range in s["sharpwave_analysis_settings"][
            "filter_ranges_hz"
        ]:
            assert isinstance(
                filter_range[0],
                int,
            ), f"filter range needs to be of type int, got {filter_range[0]}"
            assert isinstance(
                filter_range[1],
                int,
            ), f"filter range needs to be of type int, got {filter_range[1]}"
            assert (
                filter_range[1] > filter_range[0]
            ), f"second filter value needs to be higher than first one, got {filter_range}"
            assert filter_range[0] < sfreq and filter_range[1] < sfreq, (
                "filter range has to be smaller than sfreq, "
                f"got sfreq {sfreq} and filter range {filter_range}"
            )
        # check if all features are also enbled via an estimator
        used_features = list()
        for feature_name, val in s["sharpwave_analysis_settings"][
            "sharpwave_features"
        ].items():
            assert isinstance(
                val, bool
            ), f"sharpwave_feature type {feature_name} has to be of type bool, got {val}"
            if val is True:
                used_features.append(feature_name)
        for used_feature in used_features:
            estimator_list_feature = (
                []
            )  # one feature can have multiple estimators
            for estimator, est_features in s["sharpwave_analysis_settings"][
                "estimator"
            ].items():
                if est_features is not None:
                    for est_feature in est_features:
                        if used_feature == est_feature:
                            estimator_list_feature.append(estimator)
            assert (
                len(estimator_list_feature) > 0
            ), f"add estimator key for {used_feature}"
