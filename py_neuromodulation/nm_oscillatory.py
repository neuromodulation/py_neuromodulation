from scipy import fft, signal
import numpy as np
from abc import ABC
from typing import Iterable

from py_neuromodulation import nm_filter, nm_features_abc, nm_kalmanfilter


class OscillatoryFeature(nm_features_abc.Feature, ABC):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names

        self.fband_names = [
            value for value in settings["frequency_ranges_hz"].keys()
        ]

        self.f_ranges = [
            settings["frequency_ranges_hz"][fband_name]
            for fband_name in self.fband_names
        ]

        self.KF_dict = {}

        if settings["features"]["kalman_filter"] is True:
            for f_band in settings["kalman_filter_settings"]["frequency_bands"]:
                for channel in self.ch_names:
                    self.KF_dict[
                        "_".join([channel, f_band])
                    ] = nm_kalmanfilter.define_KF(
                        settings["kalman_filter_settings"]["Tp"],
                        settings["kalman_filter_settings"]["sigma_w"],
                        settings["kalman_filter_settings"]["sigma_v"],
                    )

    def update_KF(self, feature_calc: float, fband: str, ch_name: str) -> float:
        if fband in self.s["kalman_filter_settings"]["frequency_bands"]:
            KF_name = "_".join([ch_name, fband])
            self.KF_dict[KF_name].predict()
            self.KF_dict[KF_name].update(feature_calc)
            feature_calc = self.KF_dict[KF_name].x[0]  # filtered signal
        return feature_calc


class FFT(OscillatoryFeature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        super().__init__(
            settings, ch_names, sfreq
        )  # needs to be called to init KF_dict

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            data_ch = data[
                ch_idx,
                -int(
                    self.s["fft_settings"]["windowlength_ms"]
                    * self.sfreq
                    / 1000
                ) :,
            ]

            Z = np.abs(fft.rfft(data_ch))
            f = np.arange(
                0, int(self.s["fft_settings"]["windowlength_ms"] / 2) + 1, 1
            )
            for idx_fband, f_range in enumerate(self.f_ranges):
                fband = self.f_band_names[idx_fband]
                idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
                feature_calc = np.mean(Z[idx_range])

                if self.s["fft_settings"]["log_transform"]:
                    feature_calc = np.log(feature_calc)

                if self.s["fft_settings"]["kalman_filter"] is True:
                    feature_calc = self.update_KF(feature_calc, fband, ch_name)

                feature_name = "_".join([ch_name, "fft", fband])
                features_compute[feature_name] = feature_calc
        return features_compute


class STFT(OscillatoryFeature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.ch_names = ch_names
        self.s = settings
        self.sfreq = sfreq

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            f, t, Zxx = signal.stft(
                data[ch_idx, :],
                sfreq=self.sfreq,
                window="hamming",
                nperseg=int(self.s["stft_settings"]["windowlength_ms"]),
                boundary="even",
            )
            Z = np.abs(Zxx)
            for idx_fband, f_range in enumerate(self.f_ranges):
                fband = self.f_band_names[idx_fband]
                idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
                feature_calc = np.mean(Z[idx_range, :])  # 1. dim: f, 2. dim: t

                if self.s["stft_settings"]["kalman_filter"] is True:
                    feature_calc = self.update_KF(feature_calc, fband, ch_name)

                feature_name = "_".join([ch_name, "stft", fband])
                features_compute[feature_name] = feature_calc
        return features_compute


class BandPower(OscillatoryFeature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.ch_names = ch_names
        self.s = settings
        self.sfreq = sfreq

        self.seglengths = np.floor(
            self.sfreq
            / 1000
            * np.array(
                list(
                    self.s["bandpass_filter_settings"][
                        "segment_lengths_ms"
                    ].values()
                )
            )
        ).astype(int)
        self.fband_names = [
            value for value in self.s["frequency_ranges_hz"].keys()
        ]
        self.f_ranges = [
            self.s["frequency_ranges_hz"][fband_name]
            for fband_name in self.fband_names
        ]

        self.bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=self.f_ranges,
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:

        data = self.bandpass_filter.filter_data(data)
        for ch_idx, ch_name in enumerate(self.ch_names):
            for idx, f_band in enumerate(self.s["frequency_ranges_hz"].keys()):
                seglength = self.seglengths[ch_idx]
                for bp_feature in [
                    k
                    for k, v in self.s["bandpass_filter_settings"][
                        "bandpower_features"
                    ].items()
                    if v is True
                ]:
                    if bp_feature == "activity":
                        if self.s["bandpass_filter_settings"]["log_transform"]:
                            feature_calc = np.log(
                                np.var(data[ch_idx, -seglength:])
                            )
                        else:
                            feature_calc = np.var(data[ch_idx, -seglength:])
                    elif bp_feature == "mobility":
                        deriv_variance = np.var(
                            np.diff(data[ch_idx, -seglength:])
                        )
                        feature_calc = np.sqrt(
                            deriv_variance / np.var(data[ch_idx, -seglength:])
                        )
                    elif bp_feature == "complexity":
                        dat_deriv = np.diff(data[ch_idx, -seglength:])
                        deriv_variance = np.var(dat_deriv)
                        mobility = np.sqrt(
                            deriv_variance / np.var(data[ch_idx, -seglength:])
                        )
                        dat_deriv_2 = np.diff(dat_deriv)
                        dat_deriv_2_var = np.var(dat_deriv_2)
                        deriv_mobility = np.sqrt(
                            dat_deriv_2_var / deriv_variance
                        )
                        feature_calc = deriv_mobility / mobility
                    if (
                        self.s["bandpass_filter_settings"]["kalman_filter"]
                        is True
                    ) and (bp_feature == "activity"):
                        feature_calc = self.update_KF(
                            feature_calc, f_band, ch_name
                        )

                    feature_name = "_".join(
                        [ch_name, "bandpass", bp_feature, f_band]
                    )
                    features_compute[feature_name] = feature_calc
        return features_compute
