# from multiprocessing import Process, Manager
from numpy import arange, array, ceil, floor, where

from mne.filter import notch_filter

from pyneuromodulation import nm_bandpower, nm_filter, nm_hjorth_raw, nm_kalmanfilter, nm_sharpwaves, nm_coherence


class Features:

    def __init__(self, s, verbose=None) -> None:
        """Initialize Feature module

        Parameters
        ----------
        s : dict
            settings.SettingsWrapper initialized dictionary
        """

        self.s = s  # settings
        self.verbose = verbose
        self.ch_names = list(s["ch_names"])
        if s["methods"]["raw_resampling"] is True:
            self.fs = s["raw_resampling_settings"]["resample_freq"]
        else:
            self.fs = s["fs"]
        self.line_noise = s["line_noise"]
        self.seglengths = floor(
                self.fs / 1000 * array([value[1] for value in s[
                    "bandpass_filter_settings"][
                        "frequency_ranges"].values()])).astype(int)
        print("Segment lengths (ms):", self.seglengths)
        self.KF_dict = {}

        if s["methods"]["bandpass_filter"] is True:
            self.filter_fun = nm_filter.calc_band_filters(
                f_ranges=[value[0] for value in s["bandpass_filter_settings"][
                    "frequency_ranges"].values()], sfreq=self.fs,
                filter_length=self.fs - 1, verbose=self.verbose)

        if s["methods"]["kalman_filter"] is True:
            for bp_feature in [k for k, v in s["bandpass_filter_settings"][
                               "bandpower_features"].items() if v is True]:
                for f_band in s["kalman_filter_settings"]["frequency_bands"]:
                    for channel in self.ch_names:
                        self.KF_dict['_'.join([channel, bp_feature, f_band])] \
                            = nm_kalmanfilter.define_KF(
                            s["kalman_filter_settings"]["Tp"],
                            s["kalman_filter_settings"]["sigma_w"],
                            s["kalman_filter_settings"]["sigma_v"])

        if s["methods"]["sharpwave_analysis"] is True:
            self.sw_features = nm_sharpwaves.SharpwaveAnalyzer(self.s["sharpwave_analysis_settings"],
                                                               self.fs)
        self.new_dat_index = int(self.fs / self.s["sampling_rate_features"])

    def estimate_features(self, data) -> dict:
        """ Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.

        Parameters
        ----------
        data (np array) : (channels, time)

        Returns
        -------
        dat (pd Dataframe) with naming convention:
            channel_method_feature_(f_band)
        """

        # manager = Manager()
        # features_ = manager.dict() #features_ = {}
        features_ = dict()

        # notch filter data before feature estimation
        if self.s["methods"]["notch_filter"]:
            freqs = arange(
                self.line_noise, 3 * self.line_noise, self.line_noise)
            data = notch_filter(
                x=data, Fs=self.fs, trans_bandwidth=15, freqs=freqs,
                fir_design='firwin', notch_widths=3,
                filter_length=data.shape[1]-1, verbose=False,)

        if self.s["methods"]["bandpass_filter"]:
            dat_filtered = nm_filter.apply_filter(data, self.filter_fun)  # shape (bands, time)
        else:
            dat_filtered = None

        # multiprocessing approach
        '''
        job = [Process(target=self.est_ch, args=(features_, ch_idx, ch)) for ch_idx, ch in enumerate(self.ch_names)]
        _ = [p.start() for p in job]
        _ = [p.join() for p in job]
        '''

        # sequential approach
        for ch_idx in range(len(self.ch_names)):
            ch = self.ch_names[ch_idx]
            features_ = self.est_ch(features_, ch_idx, ch, dat_filtered, data)

        if self.s["methods"]["pdc"] is True or self.s["methods"]["dtf"] is True:
            for filt_idx, filt in enumerate(
                    self.s["bandpass_filter_settings"]["frequency_ranges"].keys()):
                features_ = self.est_connect(
                    features_, filt, dat_filtered[:, filt_idx, :])

        # return dict(features_) # this is necessary for multiprocessing approach
        return features_

    def est_ch(self, features_, ch_idx, ch, dat_filtered, data) -> dict:
        """Estimate features for a given channel

        Parameters
        ----------
        features_ dict : dict
            features.py output feature dict
        ch_idx : int
            channel index
        ch : str
            channel name
        dat_filtered : np.ndarray
            notch filtered and bandbass filtered data
        data : np.ndarray)
            notch filtered data

        Returns
        -------
        features_ : dict
            features.py output feature dict
        """

        if self.s["methods"]["bandpass_filter"]:
            features_ = nm_bandpower.get_bandpower_features(
                features_, self.s, self.seglengths, dat_filtered, self.KF_dict,
                ch, ch_idx)

        if self.s["methods"]["raw_hjorth"]:
            features_ = nm_hjorth_raw.get_hjorth_raw(
                features_, data[ch_idx, :], ch)

        if self.s["methods"]["return_raw"]:
            features_['_'.join([ch, 'raw'])] = data[ch_idx, -1]  # subsampling

        if self.s["methods"]["sharpwave_analysis"]:
            features_ = self.sw_features.get_sharpwave_features(
                features_, data[ch_idx, -self.new_dat_index:], ch)

        return features_

    def est_connect(self, features_, filt, dat_filt) -> dict:

        if self.s["methods"]["pdc"]:
            if filt in self.s["pdc_settings"]["frequency_ranges"].keys():
                chs = self.s["pdc_settings"]["frequency_ranges"][filt]
                ch_idx = [self.ch_names.index(ch) for ch in chs]
                dat_ = dat_filt[ch_idx]
                features_ = nm_coherence.get_pdc(
                    features_, self.s, dat_, filt, chs)

        if self.s["methods"]["dtf"]:
            if filt in self.s["dtf_settings"]["frequency_ranges"].keys():
                chs = self.s["dtf_settings"]["frequency_ranges"][filt]
                ch_idx = [self.ch_names.index(ch) for ch in chs]
                dat_ = dat_filt[ch_idx]
                features_ = nm_coherence.get_dtf(
                    features_, self.s, dat_, filt, chs)

        return features_
