from typing import Optional, Iterable

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

    @staticmethod
    def test_settings_osc(
        s: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
        osc_feature_name: str,
    ):
        assert (
            fb[0] < sfreq / 2 and fb[1] < sfreq / 2
            for fb in s["frequency_ranges_hz"].values()
        ), (
            "the frequency band ranges need to be smaller than the nyquist frequency"
            f"got sfreq = {sfreq} and fband ranges {s['frequency_ranges_hz']}"
        )

        if osc_feature_name != "bandpass_filter_settings":
            assert isinstance(
                s[osc_feature_name]["windowlength_ms"], int
            ), f"windowlength_ms needs to be type int, got {s[osc_feature_name]['windowlength_ms']}"

            assert (
                s[osc_feature_name]["windowlength_ms"]
                <= s["segment_length_features_ms"]
            ), (
                f"oscillatory feature windowlength_ms = ({s[osc_feature_name]['windowlength_ms']})"
                f"needs to be smaller than"
                f"s['segment_length_features_ms'] = {s['segment_length_features_ms']}",
            )
        else:
            for seg_length in s[osc_feature_name][
                "segment_lengths_ms"
            ].values():
                assert isinstance(
                    seg_length, int
                ), f"segment length has to be type int, got {seg_length}"
        assert isinstance(
            s[osc_feature_name]["log_transform"], bool
        ), f"log_transform needs to be type bool, got {s[osc_feature_name]['log_transform']}"

        assert isinstance(s["frequency_ranges_hz"], dict)

        assert (
            isinstance(value, list)
            for value in s["frequency_ranges_hz"].values()
        )
        assert (len(value) == 2 for value in s["frequency_ranges_hz"].values())

        assert (
            isinstance(value[0], list)
            for value in s["frequency_ranges_hz"].values()
        )

        assert (
            len(value[0]) == 2 for value in s["frequency_ranges_hz"].values()
        )

        assert (
            isinstance(value[1], (float, int))
            for value in s["frequency_ranges_hz"].values()
        )

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

    def estimate_osc_features(
        self,
        features_compute: dict,
        data: np.ndarray,
        feature_name: np.ndarray,
        est_name: str,
    ):
        for feature_est_name in list(self.s[est_name]["features"].keys()):
            if self.s[est_name]["features"][feature_est_name] is True:
                # switch case for feature_est_name
                match feature_est_name:
                    case "mean":
                        features_compute[
                            f"{feature_name}_{feature_est_name}"
                        ] = np.nanmean(data)
                    case "median":
                        features_compute[
                            f"{feature_name}_{feature_est_name}"
                        ] = np.nanmedian(data)
                    case "std":
                        features_compute[
                            f"{feature_name}_{feature_est_name}"
                        ] = np.nanstd(data)
                    case "max":
                        features_compute[
                            f"{feature_name}_{feature_est_name}"
                        ] = np.nanmax(data)

        return features_compute


class FFT(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)

        if self.s["fft_settings"]["log_transform"]:
            self.log_transform = True
        else:
            self.log_transform = False

        window_ms = self.s["fft_settings"]["windowlength_ms"]
        self.window_samples = int(-np.floor(window_ms / 1000 * sfreq))
        self.freqs = fft.rfftfreq(
            -self.window_samples, 1 / np.floor(self.sfreq)
        )

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                idx_range = np.where(
                    (self.freqs >= f_range[0]) & (self.freqs < f_range[1])
                )[0]
                feature_name = "_".join([ch_name, "fft", fband])
                self.feature_params.append((ch_idx, feature_name, idx_range))

    @staticmethod
    def test_settings(s: dict, ch_names: Iterable[str], sfreq: int | float):
        OscillatoryFeature.test_settings_osc(s, ch_names, sfreq, "fft_settings")

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = data[:, self.window_samples :]
        Z = np.abs(fft.rfft(data))

        if self.log_transform:
            Z = np.log10(Z)

        for ch_idx, feature_name, idx_range in self.feature_params:
            Z_ch = Z[ch_idx, idx_range]

            features_compute = self.estimate_osc_features(
                features_compute, Z_ch, feature_name, "fft_settings"
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.s["fft_settings"]["return_spectrum"]:
                features_compute.update(
                    {
                        f"{ch_name}_fft_psd_{str(f)}": Z[ch_idx][idx]
                        for idx, f in enumerate(self.freqs.astype(int))
                    }
                )

        return features_compute


class Welch(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)

        self.log_transform = self.s["welch_settings"]["log_transform"]

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                feature_name = "_".join([ch_name, "welch", fband])
                self.feature_params.append((ch_idx, feature_name, f_range))

    @staticmethod
    def test_settings(s: dict, ch_names: Iterable[str], sfreq: int | float):
        OscillatoryFeature.test_settings_osc(
            s, ch_names, sfreq, "welch_settings"
        )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        freqs, Z = signal.welch(
            data,
            fs=self.sfreq,
            window="hann",
            nperseg=self.sfreq,
            noverlap=None,
        )

        if self.log_transform:
            Z = np.log10(Z)

        for ch_idx, feature_name, f_range in self.feature_params:
            Z_ch = Z[ch_idx]

            idx_range = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[
                0
            ]

            features_compute = self.estimate_osc_features(
                features_compute,
                Z_ch[idx_range],
                feature_name,
                "welch_settings",
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.s["welch_settings"]["return_spectrum"]:
                features_compute.update(
                    {
                        f"{ch_name}_welch_psd_{str(f)}": Z[ch_idx][idx]
                        for idx, f in enumerate(freqs.astype(int))
                    }
                )

        return features_compute


class STFT(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)

        self.nperseg = int(self.s["stft_settings"]["windowlength_ms"])
        self.log_transform = self.s["stft_settings"]["log_transform"]

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                feature_name = "_".join([ch_name, "stft", fband])
                self.feature_params.append((ch_idx, feature_name, f_range))

    @staticmethod
    def test_settings(s: dict, ch_names: Iterable[str], sfreq: int | float):
        OscillatoryFeature.test_settings_osc(
            s, ch_names, sfreq, "stft_settings"
        )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        freqs, _, Zxx = signal.stft(
            data,
            fs=self.sfreq,
            window="hamming",
            nperseg=self.nperseg,
            boundary="even",
        )
        Z = np.abs(Zxx)
        if self.log_transform:
            Z = np.log10(Z)
        for ch_idx, feature_name, f_range in self.feature_params:
            Z_ch = Z[ch_idx]
            idx_range = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[
                0
            ]

            features_compute = self.estimate_osc_features(
                features_compute,
                Z_ch[idx_range, :],
                feature_name,
                "stft_settings",
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.s["stft_settings"]["return_spectrum"]:
                Z_ch_mean = Z[ch_idx].mean(axis=1)
                features_compute.update(
                    {
                        f"{ch_name}_stft_psd_{str(f)}": Z_ch_mean[idx]
                        for idx, f in enumerate(freqs.astype(int))
                    }
                )

        return features_compute


class BandPower(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
        use_kf: Optional[bool] = None,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        bp_settings = self.s["bandpass_filter_settings"]

        self.bandpass_filter = nm_filter.MNEFilter(
            f_ranges=list(self.f_ranges_dict.values()),
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        self.log_transform = bp_settings["log_transform"]

        if use_kf is True or (
            use_kf is None and bp_settings["kalman_filter"] is True
        ):
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

    @staticmethod
    def test_settings(s: dict, ch_names: Iterable[str], sfreq: int | float):
        OscillatoryFeature.test_settings_osc(
            s, ch_names, sfreq, "bandpass_filter_settings"
        )

        assert (
            isinstance(value, bool)
            for value in s["bandpass_filter_settings"][
                "bandpower_features"
            ].values()
        )

        assert any(
            value is True
            for value in s["bandpass_filter_settings"][
                "bandpower_features"
            ].values()
        ), "Set at least one bandpower_feature to True."

        for fband_name, seg_length_fband in s["bandpass_filter_settings"][
            "segment_lengths_ms"
        ].items():
            assert isinstance(seg_length_fband, int), (
                f"bandpass segment_lengths_ms for {fband_name} "
                f"needs to be of type int, got {seg_length_fband}"
            )

            assert seg_length_fband <= s["segment_length_features_ms"], (
                f"segment length {seg_length_fband} needs to be smaller than "
                f" s['segment_length_features_ms'] = {s['segment_length_features_ms']}"
            )

        for fband_name in list(s["frequency_ranges_hz"].keys()):
            assert fband_name in list(
                s["bandpass_filter_settings"]["segment_lengths_ms"].keys()
            ), (
                f"frequency range {fband_name} "
                "needs to be defined in s['bandpass_filter_settings']['segment_lengths_ms']"
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
                    feature_calc = np.log10(
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

            features_compute[feature_name] = np.nan_to_num(feature_calc)

        return features_compute
