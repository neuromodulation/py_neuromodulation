import numpy as np
from fooof import FOOOF
from scipy import fft

class SpectrumAnalyzer:

    def __init__(
        self,
        settings_fooof : dict,
        fs : int,
        verbose : bool
    ) -> None:

        self.fs = fs
        self.settings_fooof = settings_fooof
        self.freq_range = settings_fooof["freq_range_hz"]
        self.ap_mode = 'knee' if settings_fooof['knee'] else 'fixed'
        self.max_n_peaks = settings_fooof["max_n_peaks"]

        self.fm = FOOOF(
            aperiodic_mode=self.ap_mode,
            peak_width_limits=settings_fooof["peak_width_limits"],
            max_n_peaks=settings_fooof["max_n_peaks"],
            min_peak_height=settings_fooof["min_peak_height"],
            peak_threshold=settings_fooof["peak_threshold"],
            verbose=verbose
        )

        self.num_samples = int(
            settings_fooof["windowlength_ms"]*fs/1000
        )

        self.f_vec = np.arange(0, int(settings_fooof["windowlength_ms"]/2) + 1, 1)

    def _get_spectrum(self, data: np.array):
        """return absolute value fft spectrum"""

        data = data[-self.num_samples:]
        Z = np.abs(fft.rfft(data))

        return Z

    def get_fooof_params(self, features_ : dict, data : np.array, ch : str):

        spectrum = self._get_spectrum(data)

        self.fm.fit(
            self.f_vec,
            spectrum,
            self.freq_range
        )

        if self.settings_fooof["aperiodic"]["exponent"]:
            features_[f"{ch}_fooof_a_exp"] = self.fm.get_params(
                'aperiodic_params',
                'exponent'
            ) if self.fm.fooofed_spectrum_ is not None else None

        if self.settings_fooof["aperiodic"]["offset"]:
            features_[f"{ch}_fooof_a_offset"] = self.fm.get_params(
                'aperiodic_params',
                'offset'
            ) if self.fm.fooofed_spectrum_ is not None else None

        peaks_bw = self.fm.get_params('peak_params', 'BW') \
            if self.fm.fooofed_spectrum_ is not None else None
        peaks_cf = self.fm.get_params('peak_params', 'CF') \
            if self.fm.fooofed_spectrum_ is not None else None
        peaks_pw = self.fm.get_params('peak_params', 'PW') \
            if self.fm.fooofed_spectrum_ is not None else None

        if type(peaks_bw) is np.float64 or peaks_bw is None:
            peaks_bw = [peaks_bw]
            peaks_cf = [peaks_cf]
            peaks_pw = [peaks_pw]

        for peak_idx in range(self.max_n_peaks):

            if self.settings_fooof["periodic"]["band_width"]:
                features_[f"{ch}_fooof_p_{peak_idx}_bw"] = peaks_bw[peak_idx] \
                    if peak_idx < len(peaks_bw) else None

            if self.settings_fooof["periodic"]["center_frequency"]:
                features_[f"{ch}_fooof_p_{peak_idx}_cf"] = peaks_cf[peak_idx] \
                    if peak_idx < len(peaks_bw) else None

            if self.settings_fooof["periodic"]["height_over_ap"]:
                features_[f"{ch}_fooof_p_{peak_idx}_pw"] = peaks_pw[peak_idx] \
                    if peak_idx < len(peaks_bw) else None
 
        return features_
