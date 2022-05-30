import numpy as np
from scipy.signal import convolve, find_peaks
from mne.filter import create_filter
from typing import Iterable

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

            self.list_filter.append(
                (
                    f"range_{filter_range[0]}_{filter_range[1]}",
                    create_filter(
                        None,
                        sfreq,
                        l_freq=filter_range[0],
                        h_freq=filter_range[1],
                        fir_design="firwin",
                        l_trans_bandwidth=5,
                        h_trans_bandwidth=5,
                        filter_length=str(sfreq) + "ms",
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
        self.initialize_sw_features()

        # initializing estimator functions, respecitive for all sharpwave features
        fun_names = []
        for used_feature in self.used_features:
            for estimator, est_features in self.sw_settings[
                "estimator"
            ].items():
                if est_features is not None:
                    for est_feature in est_features:
                        if used_feature == est_feature:
                            fun_names.append(estimator)

        self.estimator_names = fun_names
        self.estimator_functions = [
            getattr(np, est_name) for est_name in self.estimator_names
        ]

    def initialize_sw_features(self) -> None:
        """Resets used attributes to empty lists"""
        for feature_name in self.used_features:
            setattr(self, feature_name, list())
        if "trough" not in self.used_features:
            # trough attribute is still necessary, even if it is not specified in settings
            self.trough = list()
        self.troughs_idx = list()

    def get_peaks_around(self, trough_ind, arr_ind_peaks, filtered_dat):
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

        ind_greater = np.where(arr_ind_peaks > trough_ind)[0]
        if ind_greater.shape[0] == 0:
            raise NoValidTroughException("No valid trough")
        val_ind_greater = arr_ind_peaks[ind_greater]
        peak_right_idx = arr_ind_peaks[
            ind_greater[np.argsort(val_ind_greater)[0]]
        ]

        ind_smaller = np.where(arr_ind_peaks < trough_ind)[0]
        if ind_smaller.shape[0] == 0:
            raise NoValidTroughException("No valid trough")

        val_ind_smaller = arr_ind_peaks[ind_smaller]
        peak_left_idx = arr_ind_peaks[
            ind_smaller[np.argsort(val_ind_smaller)[-1]]
        ]

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
    ):
        """Given a new data batch, the peaks, troughs and sharpwave features
        are estimated. Importantly only new data is being analyzed here. In
        steps of 1/settings["sampling_rate_features] are analyzed and returned.
        Data is assumed to be notch filtered and bandpass filtered beforehand.

        Parameters
        ----------
        features_ (dict): Features.py estimated features
        data (np.ndarray): 1d single channel data batch
        ch (string): channel name
        Returns
        -------
        features_ (dict): set features for Features.py object
        """
        for ch_idx, ch_name in enumerate(self.ch_names):
            for filter_name, filter in self.list_filter:
                self.filtered_data = convolve(
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

                        self.filtered_data = -self.filtered_data

                    self.initialize_sw_features()  # reset sharpwave feature attriubtes to empty lists
                    self.analyze_waveform()

                    # for each feature take the respective fun.
                    for feature_idx, feature_name in enumerate(
                        self.used_features
                    ):
                        key_name = "_".join(
                            [
                                ch_name,
                                "Sharpwave",
                                self.estimator_names[feature_idx].title(),
                                feature_name,
                                filter_name,
                            ]
                        )
                        val = (
                            self.estimator_functions[feature_idx](
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
                        features_compute[key_name] = self.estimator_functions[
                            idx
                        ](
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

        peaks = find_peaks(
            self.filtered_data,
            distance=self.sw_settings["detect_troughs"]["distance_peaks_ms"],
        )[0]
        troughs = find_peaks(
            -self.filtered_data,
            distance=self.sw_settings["detect_troughs"]["distance_troughs_ms"],
        )[0]

        for trough_idx in troughs:
            try:
                (
                    peak_idx_left,
                    peak_idx_right,
                    peak_left,
                    peak_right,
                ) = self.get_peaks_around(trough_idx, peaks, self.filtered_data)
            except NoValidTroughException:
                # in this case there are no adjacent two peaks around this trough
                # str(e) could print the exception error message
                # print(str(e))
                continue

            trough = self.filtered_data[trough_idx]
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
                    >= self.filtered_data.shape[0]
                ):
                    continue

                sharpness = (
                    (
                        self.filtered_data[trough_idx]
                        - self.filtered_data[
                            trough_idx - int(5 * (1000 / self.sfreq))
                        ]
                    )
                    + (
                        self.filtered_data[trough_idx]
                        - self.filtered_data[
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
                    np.diff(self.filtered_data[peak_idx_left : trough_idx + 1])
                )
                self.rise_steepness.append(rise_steepness)

            if (
                self.sw_settings["sharpwave_features"]["decay_steepness"]
                is True
            ):
                decay_steepness = np.max(
                    np.diff(self.filtered_data[trough_idx : peak_idx_right + 1])
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
                        - self.filtered_data[trough_idx]
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
