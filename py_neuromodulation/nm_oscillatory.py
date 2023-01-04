from typing import Iterable

import numpy as np
from scipy import fft, signal

from py_neuromodulation import nm_filter, nm_features_abc, nm_kalmanfilter


class OscillatoryFeature(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict = {}

        self.f_ranges_dict = settings["frequency_ranges_hz"]
        self.fband_names = list(settings["frequency_ranges_hz"].keys())
        self.f_ranges = list(settings["frequency_ranges_hz"].values())

    def init_KF(self, feature: str) -> None:
        for f_band in self.s["kalman_filter_settings"]["frequency_bands"]:
            for channel in self.ch_names:
                self.KF_dict[
                    "_".join([channel, feature, f_band])
                ] = nm_kalmanfilter.define_KF(
                    self.s["kalman_filter_settings"]["Tp"],
                    self.s["kalman_filter_settings"]["sigma_w"],
                    self.s["kalman_filter_settings"]["sigma_v"],
                )

    def update_KF(self, feature_calc: float, KF_name: str) -> float:
        if KF_name in self.KF_dict:
            self.KF_dict[KF_name].predict()
            self.KF_dict[KF_name].update(feature_calc)
            feature_calc = self.KF_dict[KF_name].x[0]
        return feature_calc


class FFT(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        if self.s["fft_settings"]["kalman_filter"]:
            self.init_KF("fft")

        if self.s["fft_settings"]["log_transform"]:
            self.log_transform = True
        else:
            self.log_transform = False

        window_ms = self.s["fft_settings"]["windowlength_ms"]
        self.window_samples = int(-np.floor(window_ms / 1000 * sfreq))
        freqs = fft.rfftfreq(-self.window_samples, 1 / np.floor(self.sfreq))

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                idx_range = np.where(
                    (freqs >= f_range[0]) & (freqs < f_range[1])
                )[0]
                feature_name = "_".join([ch_name, "fft", fband])
                self.feature_params.append((ch_idx, feature_name, idx_range))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = data[:, self.window_samples :]
        Z = np.abs(fft.rfft(data))
        for ch_idx, feature_name, idx_range in self.feature_params:
            Z_ch = Z[ch_idx, idx_range]
            feature_calc = np.mean(Z_ch)

            if self.log_transform:
                feature_calc = np.log(feature_calc)

            if self.KF_dict:
                feature_calc = self.update_KF(feature_calc, feature_name)

            features_compute[feature_name] = feature_calc
        return features_compute


class STFT(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        if self.s["stft_settings"]["kalman_filter"]:
            self.init_KF("stft")

        self.nperseg = int(self.s["stft_settings"]["windowlength_ms"])

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                feature_name = "_".join([ch_name, "stft", fband])
                self.feature_params.append((ch_idx, feature_name, f_range))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        f, _, Zxx = signal.stft(
            data,
            fs=self.sfreq,
            window="hamming",
            nperseg=self.nperseg,
            boundary="even",
        )
        Z = np.abs(Zxx)
        for ch_idx, feature_name, f_range in self.feature_params:
            Z_ch = Z[ch_idx]
            idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
            feature_calc = np.mean(Z_ch[idx_range, :])  # 1. dim: f, 2. dim: t

            if self.KF_dict:
                feature_calc = self.update_KF(feature_calc, feature_name)

            features_compute[feature_name] = feature_calc

        return features_compute


class BandPower(OscillatoryFeature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        bp_settings = self.s["bandpass_filter_settings"]

        self.bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=list(self.f_ranges_dict.values()),
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        self.log_transform = bp_settings["log_transform"]

        if bp_settings["kalman_filter"]:
            self.init_KF("bandpass_activity")

        bp_features = ["activity", "mobility", "complexity"]
        seglengths = bp_settings["segment_lengths_ms"]

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for f_band_idx, f_band in enumerate(self.f_ranges_dict):
                seglength_ms = seglengths[f_band]
                seglen = int(np.floor(self.sfreq / 1000 * seglength_ms))
                for bp_feature, v in bp_settings["bandpower_features"].items():
                    if v is True:
                        if bp_feature not in bp_features:
                            raise ValueError()
                        feature_name = "_".join(
                            [ch_name, "bandpass", bp_feature, f_band]
                        )
                        self.feature_params.append(
                            (
                                ch_idx,
                                ch_name,
                                f_band,
                                f_band_idx,
                                seglen,
                                bp_feature,
                                feature_name,
                            )
                        )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:

        data = self.bandpass_filter.filter_data(data)

        for (
            ch_idx,
            ch_name,
            f_band,
            f_band_idx,
            seglen,
            bp_feature,
            feature_name,
        ) in self.feature_params:
            if bp_feature == "activity":
                if self.log_transform:
                    feature_calc = np.log(
                        np.var(data[ch_idx, f_band_idx, -seglen:])
                    )
                else:
                    feature_calc = np.var(data[ch_idx, f_band_idx, -seglen:])
            elif bp_feature == "mobility":
                deriv_variance = np.var(
                    np.diff(data[ch_idx, f_band_idx, -seglen:])
                )
                feature_calc = np.sqrt(
                    deriv_variance / np.var(data[ch_idx, f_band_idx, -seglen:])
                )
            elif bp_feature == "complexity":
                dat_deriv = np.diff(data[ch_idx, f_band_idx, -seglen:])
                deriv_variance = np.var(dat_deriv)
                mobility = np.sqrt(
                    deriv_variance / np.var(data[ch_idx, f_band_idx, -seglen:])
                )
                dat_deriv_2 = np.diff(dat_deriv)
                dat_deriv_2_var = np.var(dat_deriv_2)
                deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)
                feature_calc = deriv_mobility / mobility

            if self.KF_dict and (bp_feature == "activity"):
                feature_calc = self.update_KF(feature_calc, feature_name)

            features_compute[feature_name] = feature_calc

        return features_compute
