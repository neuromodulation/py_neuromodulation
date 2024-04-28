from collections.abc import Iterable
import numpy as np
from fooof import FOOOF
from scipy import fft

from py_neuromodulation.nm_features_abc import Feature
from py_neuromodulation import logger


class FooofAnalyzer(Feature):
    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.settings_fooof = settings["fooof"]
        self.sfreq = sfreq
        self.ch_names = ch_names

        self.freq_range = self.settings_fooof["freq_range_hz"]
        self.ap_mode = "knee" if self.settings_fooof["knee"] else "fixed"
        self.max_n_peaks = self.settings_fooof["max_n_peaks"]

        self.num_samples = int(self.settings_fooof["windowlength_ms"] * sfreq / 1000)

        self.f_vec = np.arange(0, int(self.num_samples / 2) + 1, 1)

    @staticmethod
    def test_settings(
        s: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ):
        assert isinstance(s["fooof"]["aperiodic"]["exponent"], bool)
        assert isinstance(s["fooof"]["aperiodic"]["offset"], bool)
        assert isinstance(s["fooof"]["aperiodic"]["knee"], bool)
        assert isinstance(s["fooof"]["periodic"]["center_frequency"], bool)
        assert isinstance(s["fooof"]["periodic"]["band_width"], bool)
        assert isinstance(s["fooof"]["periodic"]["height_over_ap"], bool)
        assert isinstance(s["fooof"]["knee"], bool)
        assert isinstance(s["fooof"]["windowlength_ms"], (int, float))
        assert s["fooof"]["windowlength_ms"] <= s["segment_length_features_ms"], (
            "fooof windowlength_ms needs to be smaller equal than segment_length_features_ms "
            f"got windowlength_ms: {s['fooof']['windowlength_ms']} and {s['segment_length_features_ms']}"
        )

        assert (
            s["fooof"]["freq_range_hz"][0] < sfreq
            and s["fooof"]["freq_range_hz"][1] < sfreq
        ), f"fooof frequency range needs to be below sfreq, got {s['fooof']['freq_range_hz']}"

    def _get_spectrum(self, data: np.ndarray):
        """return absolute value fft spectrum"""

        data = data[-self.num_samples :]
        Z = np.abs(fft.rfft(data))

        return Z

    def calc_feature(
        self,
        data: np.ndarray,
        features_compute: dict,
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            spectrum = self._get_spectrum(data[ch_idx, :])

            try:
                fm = FOOOF(
                    aperiodic_mode=self.ap_mode,
                    peak_width_limits=self.settings_fooof["peak_width_limits"],
                    max_n_peaks=self.settings_fooof["max_n_peaks"],
                    min_peak_height=self.settings_fooof["min_peak_height"],
                    peak_threshold=self.settings_fooof["peak_threshold"],
                    verbose=False,
                )
                fm.fit(self.f_vec, spectrum, self.freq_range)
            except Exception as e:
                logger.critical(e, exc_info=True)

            if fm.fooofed_spectrum_ is None:
                FIT_PASSED = False
            else:
                FIT_PASSED = True

            if self.settings_fooof["aperiodic"]["exponent"]:
                features_compute[f"{ch_name}_fooof_a_exp"] = (
                    np.nan_to_num(fm.get_params("aperiodic_params", "exponent"))
                    if FIT_PASSED is True
                    else None
                )

            if self.settings_fooof["aperiodic"]["offset"]:
                features_compute[f"{ch_name}_fooof_a_offset"] = (
                    np.nan_to_num(fm.get_params("aperiodic_params", "offset"))
                    if FIT_PASSED is True
                    else None
                )

            if self.settings_fooof["aperiodic"]["knee"]:
                if not FIT_PASSED:
                    knee_freq = None
                else:
                    if fm.get_params("aperiodic_params", "exponent") != 0:
                        knee_fooof = fm.get_params("aperiodic_params", "knee")
                        knee_freq = np.nan_to_num(
                            knee_fooof
                            ** (1 / fm.get_params("aperiodic_params", "exponent"))
                        )
                    else:
                        knee_freq = None

                features_compute[f"{ch_name}_fooof_a_knee_frequency"] = knee_freq

            peaks_bw = (
                fm.get_params("peak_params", "BW") if FIT_PASSED is True else None
            )
            peaks_cf = (
                fm.get_params("peak_params", "CF") if FIT_PASSED is True else None
            )
            peaks_pw = (
                fm.get_params("peak_params", "PW") if FIT_PASSED is True else None
            )

            if type(peaks_bw) is np.float64 or peaks_bw is None:
                peaks_bw = [peaks_bw]
                peaks_cf = [peaks_cf]
                peaks_pw = [peaks_pw]

            for peak_idx in range(self.max_n_peaks):
                if self.settings_fooof["periodic"]["band_width"]:
                    features_compute[f"{ch_name}_fooof_p_{peak_idx}_bw"] = (
                        peaks_bw[peak_idx] if peak_idx < len(peaks_bw) else None
                    )

                if self.settings_fooof["periodic"]["center_frequency"]:
                    features_compute[f"{ch_name}_fooof_p_{peak_idx}_cf"] = (
                        peaks_cf[peak_idx] if peak_idx < len(peaks_bw) else None
                    )

                if self.settings_fooof["periodic"]["height_over_ap"]:
                    features_compute[f"{ch_name}_fooof_p_{peak_idx}_pw"] = (
                        peaks_pw[peak_idx] if peak_idx < len(peaks_bw) else None
                    )

        return features_compute
