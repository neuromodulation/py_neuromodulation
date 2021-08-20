from numpy import argsort, diff, max, where
from numpy import abs as np_abs
from numpy import mean as np_mean
from numpy import median as np_median
from numpy import var as np_var
from numpy import min as np_min
from numpy import max as np_max
from scipy.signal import convolve, find_peaks
from mne.filter import create_filter


class NoValidTroughException(Exception):
    pass


class SharpwaveAnalyzer:

    def __init__(self, sw_settings, sfreq) -> None:
        """
        Parameters
        ----------
        sw_settings : dict
            Sharpwave settings from settings.json
        sfreq : float
            data sampling frequency
        """

        self.sw_settings = sw_settings
        self.sfreq = sfreq
        self.filter = \
            create_filter(None, sfreq,
                          l_freq=sw_settings['filter_low_cutoff'],
                          h_freq=sw_settings['filter_high_cutoff'],
                          fir_design='firwin', l_trans_bandwidth=4,
                          h_trans_bandwidth=4, filter_length=str(sfreq)+'ms',
                          verbose=False)

        # initialize used features
        self.used_features = list()
        for feature_name, val in self.sw_settings["sharpwave_features"].items():
            if val is True:
                self.used_features.append(feature_name)

        # initialize attributes
        self.initialize_sw_features()

    def initialize_sw_features(self) -> None:
        """Resets used attributes to empty lists
        """
        for feature_name in self.used_features:
            setattr(self, feature_name, list())
        self.troughs_idx = list()

    def get_peaks_around(self, trough_ind, arr_ind_peaks, filtered_dat):
        """ Find the closest peaks to the right and left side a given trough.

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

        ind_greater = where(arr_ind_peaks > trough_ind)[0]
        if ind_greater.shape[0] == 0:
            raise NoValidTroughException("No valid trough")
        val_ind_greater = arr_ind_peaks[ind_greater]
        peak_right_idx = arr_ind_peaks[ind_greater[argsort(val_ind_greater)[0]]]

        ind_smaller = where(arr_ind_peaks < trough_ind)[0]
        if ind_smaller.shape[0] == 0:
            raise NoValidTroughException("No valid trough")

        val_ind_smaller = arr_ind_peaks[ind_smaller]
        peak_left_idx = \
            arr_ind_peaks[ind_smaller[argsort(val_ind_smaller)[-1]]]

        return peak_left_idx, peak_right_idx, filtered_dat[peak_left_idx], \
            filtered_dat[peak_right_idx]

    def get_sharpwave_features(self, features_, data, ch):
        """ Given a new data batch, the peaks, troughs and sharpwave features
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
        self.filtered_data = convolve(data, self.filter, mode='same')

        # check settings if troughs and peaks are analyzed

        for detect_troughs in [True, False]:

            if detect_troughs is False:
                if self.sw_settings["detect_peaks"]["estimate"] is False:
                    continue
                key_name = 'Peak'
                # the detect_troughs loop start with peaks, s.t. data does not
                # need to be flipped

            if detect_troughs is True:
                if self.sw_settings["detect_troughs"]["estimate"] is False:
                    continue
                key_name = 'Trough'

                self.filtered_data = -self.filtered_data

            self.initialize_sw_features()  # reset sharpwave feature attriubtes to empty lists
            self.analyze_waveform()

            if self.sw_settings["estimator"]["mean"] is True:
                for feature_name in self.used_features:
                    features_['_'.join([ch, 'Sharpwave', 'Mean', key_name, feature_name])] = \
                        np_mean(getattr(self, feature_name)) if len(getattr(self, feature_name)) != 0 else 0
            if self.sw_settings["estimator"]["var"] is True:
                for feature_name in self.used_features:
                    features_['_'.join([ch, 'Sharpwave', 'Var', key_name, feature_name])] = \
                        np_var(getattr(self, feature_name)) if len(getattr(self, feature_name)) != 0 else 0
            if self.sw_settings["estimator"]["median"] is True:
                for feature_name in self.used_features:
                    features_['_'.join([ch, 'Sharpwave', 'Median', key_name, feature_name])] = \
                        np_median(getattr(self, feature_name)) if len(getattr(self, feature_name)) != 0 else 0
            if self.sw_settings["estimator"]["min"] is True:
                for feature_name in self.used_features:
                    features_['_'.join([ch, 'Sharpwave', 'Min', key_name, feature_name])] = \
                        np_min(getattr(self, feature_name)) if len(getattr(self, feature_name)) != 0 else 0
            if self.sw_settings["estimator"]["max"] is True:
                for feature_name in self.used_features:
                    features_['_'.join([ch, 'Sharpwave', 'Max', key_name, feature_name])] = \
                        np_max(getattr(self, feature_name)) if len(getattr(self, feature_name)) != 0 else 0

        return features_

    def analyze_waveform(self) -> None:
        """ Given the scipy.signal.find_peaks  trough/peak distance
        settings specified sharpwave features are estimated.

        Parameters
        ----------

        Raises:
            NoValidTroughException: Return if no adjacent peak can be found
            NoValidTroughException: Return if no adjacent peak can be found

        """

        peaks = find_peaks(self.filtered_data, distance=self.sw_settings["detect_troughs"]["distance_peaks"])[0]
        troughs = find_peaks(-self.filtered_data, distance=self.sw_settings["detect_troughs"]["distance_troughs"])[0]

        for trough_idx in troughs:
            try:
                peak_idx_left, peak_idx_right, peak_left, peak_right = \
                    self.get_peaks_around(trough_idx, peaks, self.filtered_data)
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

                    interval = (trough_idx - self.troughs_idx[-2]) * \
                                (1000/self.sfreq)
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
                if (trough_idx - int(5*(1000/self.sfreq)) <= 0) or \
                   (trough_idx + int(5*(1000/self.sfreq)) >=
                   self.filtered_data.shape[0]):
                    continue

                sharpness = ((self.filtered_data[trough_idx] -
                             self.filtered_data[trough_idx-int(5*(1000/self.sfreq))]) +
                             (self.filtered_data[trough_idx] -
                              self.filtered_data[trough_idx+int(5*(1000/self.sfreq))])) / 2

                self.sharpness.append(sharpness)

            if self.sw_settings["sharpwave_features"]["rise_steepness"] is True:
                # steepness is calculated as the first derivative
                # from peak/trough to trough/peak
                # here  + 1 due to python syntax, s.t. the last element is included
                rise_steepness = max(diff(self.filtered_data[peak_idx_left: trough_idx+1]))
                self.rise_steepness.append(rise_steepness)

            if self.sw_settings["sharpwave_features"]["decay_steepness"] is True:
                decay_steepness = max(diff(self.filtered_data[trough_idx: peak_idx_right+1]))
                self.decay_steepness.append(decay_steepness)

            if self.sw_settings["sharpwave_features"]["rise_steepness"] is True and \
               self.sw_settings["sharpwave_features"]["decay_steepness"] is True and \
               self.sw_settings["sharpwave_features"]["slope_ratio"] is True:
                self.slope_ratio.append(rise_steepness - decay_steepness)

            if self.sw_settings["sharpwave_features"]["prominence"] is True:
                self.prominence.append(np_abs(
                    (peak_right + peak_left) / 2 - self.filtered_data[trough_idx]))  # mV

            if self.sw_settings["sharpwave_features"]["decay_time"] is True:
                self.decay_time.append((peak_idx_left - trough_idx) * (1000/self.sfreq))  # ms

            if self.sw_settings["sharpwave_features"]["rise_time"] is True:
                self.rise_time.append((peak_idx_right - trough_idx) * (1000/self.sfreq))  # ms

            if self.sw_settings["sharpwave_features"]["width"] is True:
                self.width.append(peak_idx_right - peak_idx_left)  # ms
