import numpy as np
from scipy import signal
from mne.filter import create_filter
from typing import Iterable

from py_neuromodulation.nm_features import NMFeature


class NoValidTroughException(Exception):
    pass


class SharpwaveAnalyzer(NMFeature):
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
                self.data_process_sw = (data[ch_idx, :] 
                    if filter_name == "no_filter" 
                    else signal.fftconvolve(data[ch_idx, :], filter, mode="same")
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

        """ Find left and right peak indexes for each trough """
        peak_pointer = 0
        peak_idx_left = []
        peak_idx_right = []
        first_valid = last_valid = 0

        for i, trough_idx in enumerate(troughs):
            
            # Locate peak right of current trough
            while peak_pointer < peaks.size and peaks[peak_pointer] < trough_idx:
                peak_pointer += 1

            if peak_pointer - 1 < 0: 
                # If trough has no peak to it's left, it's not valid 
                first_valid = i + 1 # Try with next one
                continue

            if peak_pointer == peaks.size:
                # If we went past the end of the peaks list, trough had no peak to its right
                continue

            last_valid = i
            peak_idx_left.append(peaks[peak_pointer - 1])
            peak_idx_right.append(peaks[peak_pointer])

        troughs = troughs[first_valid:last_valid + 1] # Remove non valid troughs
        
        peak_idx_left = np.array(peak_idx_left, dtype=np.integer)
        peak_idx_right = np.array(peak_idx_right, dtype=np.integer)

        peak_left = self.data_process_sw[peak_idx_left]
        peak_right = self.data_process_sw[peak_idx_right]
        trough_values = self.data_process_sw[troughs]

        # No need to store trough data as it is not used anywhere else in the program
        # self.trough.append(trough)
        # self.troughs_idx.append(trough_idx)
         
        """ Calculate features (vectorized) """
        
        if self.sw_settings["sharpwave_features"]["interval"]:
            self.interval = np.concatenate(([0], np.diff(troughs))) * (1000 / self.sfreq)

        if self.sw_settings["sharpwave_features"]["peak_left"]:
            self.peak_left = peak_left

        if self.sw_settings["sharpwave_features"]["peak_right"]:
            self.peak_right = peak_right

        if self.sw_settings["sharpwave_features"]["sharpness"]:
            # sharpess is calculated on a +- 5 ms window
            # valid troughs need 5 ms of margin on both siddes
            troughs_valid = troughs[np.logical_and(
                                troughs - int(5 * (1000 / self.sfreq)) > 0, 
                                troughs + int(5 * (1000 / self.sfreq)) < self.data_process_sw.shape[0])]

            self.sharpness = (
                        (self.data_process_sw[troughs_valid] - self.data_process_sw[troughs_valid - int(5 * (1000 / self.sfreq))]) +
                        (self.data_process_sw[troughs_valid] - self.data_process_sw[troughs_valid + int(5 * (1000 / self.sfreq))])
                        ) / 2

        if (self.sw_settings["sharpwave_features"]["rise_steepness"] or
            self.sw_settings["sharpwave_features"]["decay_steepness"]):
            
            # steepness is calculated as the first derivative
            steepness = np.concatenate(([0],np.diff(self.data_process_sw)))

            if self.sw_settings["sharpwave_features"]["rise_steepness"]: # left peak -> trough
                # + 1 due to python syntax, s.t. the last element is included
                self.rise_steepness = np.array([
                    np.max(np.abs(steepness[peak_idx_left[i] : troughs[i] + 1]))
                    for i in range(trough_idx.size)
                ])
                
            if self.sw_settings["sharpwave_features"]["decay_steepness"]: # trough -> right peak
                self.decay_steepness = np.array([
                    np.max(np.abs(steepness[troughs[i] : peak_idx_right[i] + 1]))
                    for i in range(trough_idx.size)
                ])

            if (self.sw_settings["sharpwave_features"]["rise_steepness"] and
                self.sw_settings["sharpwave_features"]["decay_steepness"] and
                self.sw_settings["sharpwave_features"]["slope_ratio"]):
                self.slope_ratio = self.rise_steepness - self.decay_steepness

        if self.sw_settings["sharpwave_features"]["prominence"]:
            self.prominence = np.abs((peak_right + peak_left) / 2 - trough_values)

        if self.sw_settings["sharpwave_features"]["decay_time"]:
            self.decay_time = (peak_idx_left - troughs) * (1000 / self.sfreq) # ms

        if self.sw_settings["sharpwave_features"]["rise_time"]:
            self.rise_time = (peak_idx_right - troughs) * (1000 / self.sfreq) # ms

        if self.sw_settings["sharpwave_features"]["width"]:
            self.width = peak_idx_right - peak_idx_left  # ms
    
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
