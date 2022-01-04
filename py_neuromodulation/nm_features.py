# from multiprocessing import Process, Manager
from numpy import array, floor

from py_neuromodulation import nm_bandpower, nm_filter, nm_hjorth_raw, nm_kalmanfilter, nm_sharpwaves, nm_coherence,\
    nm_stft, nm_fft


class Features:

    def __init__(self, s, ch_names, fs, line_noise, verbose=None) -> None:
        """Initialize Feature module

        Parameters
        ----------
        s : dict
            settings.SettingsWrapper initialized dictionary
        """

        self.s = s  # settings
        self.verbose = verbose
        self.ch_names = ch_names
        if s["methods"]["raw_resampling"] is True:
            self.fs = s["raw_resampling_settings"]["resample_freq"]
        else:
            self.fs = fs
        self.line_noise = line_noise
        self.fband_names = [value for value in s["frequency_ranges"].keys()]
        self.f_ranges = [self.s["frequency_ranges"][fband_name]
                         for fband_name in self.fband_names]

        self.KF_dict = {}

        if s["methods"]["bandpass_filter"] is True:
            self.seglengths = floor(
                self.fs / 1000 * array([value for value in s[
                        "bandpass_filter_settings"]["segment_lengths"].values()])).astype(int)
            self.filter_fun = nm_filter.calc_band_filters(
                f_ranges=self.f_ranges, sfreq=self.fs,
                filter_length=self.fs - 1, verbose=self.verbose)

        if s["methods"]["kalman_filter"] is True:
            for f_band in s["kalman_filter_settings"]["frequency_bands"]:
                for channel in self.ch_names:
                    self.KF_dict['_'.join([channel, f_band])] \
                        = nm_kalmanfilter.define_KF(
                        s["kalman_filter_settings"]["Tp"],
                        s["kalman_filter_settings"]["sigma_w"],
                        s["kalman_filter_settings"]["sigma_v"])

        if s["methods"]["sharpwave_analysis"] is True:
            self.sw_features = nm_sharpwaves.SharpwaveAnalyzer(self.s["sharpwave_analysis_settings"],
                                                               self.fs)
        
        if s["methods"]["coherence"] is True:
            self.coherence_objects = []
            for idx_coh in range(len(s["coherence"]["channels"])):
                fband_names = s["coherence"]["frequency_bands"][idx_coh]
                fband_specs = []
                for band_name in fband_names:
                    fband_specs.append(s["frequency_ranges"][band_name])

                ch_1_name = s["coherence"]["channels"][idx_coh][0]
                ch_1_name_reref = [ch for ch in self.ch_names if ch.startswith(ch_1_name)][0]
                ch_1_idx = self.ch_names.index(ch_1_name_reref)

                ch_2_name = s["coherence"]["channels"][idx_coh][1]
                ch_2_name_reref = [ch for ch in self.ch_names if ch.startswith(ch_2_name)][0]
                ch_2_idx = self.ch_names.index(ch_2_name_reref)

                self.coherence_objects.append(
                    nm_coherence.NM_Coherence(self.fs, self.s["coherence"]["params"][idx_coh]["window"],
                        fband_specs, fband_names, ch_1_name, ch_2_name, ch_1_idx, ch_2_idx,
                        s["coherence"]["method"][idx_coh]["coh"],
                        s["coherence"]["method"][idx_coh]["icoh"])
                )

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

        features_ = dict()

        if self.s["methods"]["bandpass_filter"]:
            dat_filtered = nm_filter.apply_filter(data, self.filter_fun)  # shape (bands, time)
        else:
            dat_filtered = None

        # sequential approach
        for ch_idx in range(len(self.ch_names)):
            ch = self.ch_names[ch_idx]
            features_ = self.est_ch(features_, ch_idx, ch, dat_filtered, data)

        if self.s["methods"]["coherence"]:
            for coh_obj in self.coherence_objects:
                features_ = coh_obj.get_coh(features_, data[coh_obj.ch_1_idx, :], data[coh_obj.ch_2_idx, :])

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

        if self.s["methods"]["stft"] is True:
            features_ = nm_stft.get_stft_features(features_, self.s, self.fs, data[ch_idx, :], self.KF_dict, ch,
                                                  self.f_ranges, self.fband_names)

        if self.s["methods"]["fft"] is True:
            features_ = nm_fft.get_fft_features(features_, self.s, self.fs, data[ch_idx, :], self.KF_dict, ch,
                                                  self.f_ranges, self.fband_names)

        return features_
