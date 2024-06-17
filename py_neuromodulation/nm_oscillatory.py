from collections.abc import Iterable
import numpy as np
from itertools import product

from py_neuromodulation.nm_types import NMBaseModel
from pydantic import field_validator
from typing import TYPE_CHECKING

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import BoolSelector

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings
    from py_neuromodulation.nm_kalmanfilter import KalmanSettings


class OscillatoryFeatures(BoolSelector):
    mean: bool = True
    median: bool = False
    std: bool = False
    max: bool = False


class OscillatorySettings(NMBaseModel):
    windowlength_ms: int = 1000
    log_transform: bool = True
    features: OscillatoryFeatures = OscillatoryFeatures(
        mean=True, median=False, std=False, max=False
    )
    return_spectrum: bool = False


ESTIMATOR_DICT = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "std": np.nanstd,
    "max": np.nanmax,
}


class OscillatoryFeature(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: int
    ) -> None:
        settings.validate()
        self.settings: OscillatorySettings  # Assignment in subclass __init__
        self.osc_feature_name: str  # Required for output

        self.sfreq = int(sfreq)
        self.ch_names = ch_names

        self.frequency_ranges = settings.frequency_ranges_hz

        # Test settings
        assert self.settings.windowlength_ms <= settings.segment_length_features_ms, (
            f"oscillatory feature windowlength_ms = ({self.settings.windowlength_ms})"
            f"needs to be smaller than"
            f"settings['segment_length_features_ms'] = {settings.segment_length_features_ms}",
        )


class FFT(OscillatoryFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: int,
    ) -> None:
        from scipy.fft import rfftfreq

        self.osc_feature_name = "fft"
        self.settings = settings.fft_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        window_ms = self.settings.windowlength_ms

        self.window_samples = int(-np.floor(window_ms / 1000 * sfreq))
        self.freqs = rfftfreq(-self.window_samples, 1 / np.floor(self.sfreq))

        # Pre-calculate frequency ranges
        self.idx_range = [
            (
                f_band,
                np.where((self.freqs >= f_range[0]) & (self.freqs < f_range[1]))[0],
            )
            for f_band, f_range in self.frequency_ranges.items()
        ]

        self.estimators = [
            (est, ESTIMATOR_DICT[est]) for est in self.settings.features.get_enabled()
        ]

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = data[:, self.window_samples :]

        from scipy.fft import rfft

        Z = np.abs(rfft(data))  # type: ignore

        if self.settings.log_transform:
            Z = np.log10(Z)

        for f_band_name, idx_range in self.idx_range:
            # TODO Can we get rid of this for-loop? Hard to vectorize windows of different lengths...
            Z_band = Z[:, idx_range]  # Data for all channels

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=1)

                for ch_idx, ch_name in enumerate(self.ch_names):
                    features_compute[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                features_compute[f"{ch_name}_fft_psd_{int(f)}"] = Z[ch_idx][idx]

        return features_compute


class Welch(OscillatoryFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: int,
    ) -> None:
        from scipy.fft import rfftfreq

        self.osc_feature_name = "welch"
        self.settings = settings.welch_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        self.freqs = rfftfreq(self.sfreq, 1 / self.sfreq)

        self.idx_range = [
            (
                f_band,
                np.where((self.freqs >= f_range[0]) & (self.freqs < f_range[1]))[0],
            )
            for f_band, f_range in self.frequency_ranges.items()
        ]

        self.estimators = [
            (est, ESTIMATOR_DICT[est]) for est in self.settings.features.get_enabled()
        ]

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import welch

        _, Z = welch(
            data,
            fs=self.sfreq,
            window="hann",
            nperseg=self.sfreq,
            noverlap=None,
        )

        if self.settings.log_transform:
            Z = np.log10(Z)

        for f_band_name, idx_range in self.idx_range:
            Z_band = Z[:, idx_range]

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=1)

                for ch_idx, ch_name in enumerate(self.ch_names):
                    features_compute[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                features_compute[f"{ch_name}_welch_psd_{str(f)}"] = Z[ch_idx][idx]

        return features_compute


class STFT(OscillatoryFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: int,
    ) -> None:
        from scipy.fft import rfftfreq

        self.osc_feature_name = "stft"
        self.settings = settings.stft_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        self.nperseg = self.settings.windowlength_ms

        self.freqs = rfftfreq(self.nperseg, 1 / self.sfreq)

        self.idx_range = [
            (
                f_band,
                np.where((self.freqs >= f_range[0]) & (self.freqs <= f_range[1]))[0],
            )
            for f_band, f_range in self.frequency_ranges.items()
        ]

        self.estimators = [
            (est, ESTIMATOR_DICT[est]) for est in self.settings.features.get_enabled()
        ]

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import stft

        _, _, Zxx = stft(
            data,
            fs=self.sfreq,
            window="hamming",
            nperseg=self.nperseg,
            boundary="even",
        )

        Z = np.abs(Zxx)
        if self.settings.log_transform:
            Z = np.log10(Z)

        for f_band_name, idx_range in self.idx_range:
            Z_band = Z[:, idx_range, :]

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=(1, 2))

                for ch_idx, ch_name in enumerate(self.ch_names):
                    features_compute[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                features_compute[f"{ch_name}_stft_psd_{str(f)}"] = Z[ch_idx].mean(
                    axis=1
                )[idx]

        return features_compute


class BandpowerFeatures(BoolSelector):
    activity: bool = True
    mobility: bool = False
    complexity: bool = False


###################################
######## BANDPOWER FEATURE ########
###################################


class BandpassSettings(NMBaseModel):
    segment_lengths_ms: dict[str, int] = {
        "theta": 1000,
        "alpha": 500,
        "low beta": 333,
        "high beta": 333,
        "low gamma": 100,
        "high gamma": 100,
        "HFA": 100,
    }
    bandpower_features: BandpowerFeatures = BandpowerFeatures()
    log_transform: bool = True
    kalman_filter: bool = False

    @field_validator("bandpower_features")
    @classmethod
    def bandpower_features_validator(cls, bandpower_features: BandpowerFeatures):
        assert (
            len(bandpower_features.get_enabled()) > 0
        ), "Set at least one bandpower_feature to True."

        return bandpower_features

    def validate_fbands(self, settings: "NMSettings") -> None:
        for fband_name, seg_length_fband in self.segment_lengths_ms.items():
            assert seg_length_fband <= settings.segment_length_features_ms, (
                f"segment length {seg_length_fband} needs to be smaller than "
                f" settings['segment_length_features_ms'] = {settings.segment_length_features_ms}"
            )

        for fband_name in settings.frequency_ranges_hz.keys():
            assert fband_name in self.segment_lengths_ms, (
                f"frequency range {fband_name} "
                "needs to be defined in settings.bandpass_filter_settings.segment_lengths_ms]"
            )


class BandPower(NMFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: float,
        use_kf: bool | None = None,
    ) -> None:
        settings.validate()

        self.bp_settings: BandpassSettings = settings.bandpass_filter_settings
        self.kalman_filter_settings: KalmanSettings = settings.kalman_filter_settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict: dict = {}

        from py_neuromodulation.nm_filter import MNEFilter

        self.bandpass_filter = MNEFilter(
            f_ranges=[
                tuple(frange) for frange in settings.frequency_ranges_hz.values()
            ],
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        if use_kf or (use_kf is None and self.bp_settings.kalman_filter):
            self.init_KF("bandpass_activity")

        seglengths = self.bp_settings.segment_lengths_ms

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for f_band_idx, f_band in enumerate(settings.frequency_ranges_hz.keys()):
                seglength_ms = seglengths[f_band]
                seglen = int(np.floor(self.sfreq / 1000 * seglength_ms))
                for bp_feature in self.bp_settings.bandpower_features.get_enabled():
                    feature_name = "_".join([ch_name, "bandpass", bp_feature, f_band])
                    self.feature_params.append(
                        (
                            ch_idx,
                            f_band_idx,
                            seglen,
                            bp_feature,
                            feature_name,
                        )
                    )

    def init_KF(self, feature: str) -> None:
        from py_neuromodulation.nm_kalmanfilter import define_KF

        for f_band in self.kalman_filter_settings.frequency_bands:
            for channel in self.ch_names:
                self.KF_dict["_".join([channel, feature, f_band])] = define_KF(
                    self.kalman_filter_settings.Tp,
                    self.kalman_filter_settings.sigma_w,
                    self.kalman_filter_settings.sigma_v,
                )

    def update_KF(self, feature_calc: np.floating, KF_name: str) -> np.floating:
        if KF_name in self.KF_dict:
            self.KF_dict[KF_name].predict()
            self.KF_dict[KF_name].update(feature_calc)
            feature_calc = self.KF_dict[KF_name].x[0]
        return feature_calc

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = self.bandpass_filter.filter_data(data)

        for (
            ch_idx,
            f_band_idx,
            seglen,
            bp_feature,
            feature_name,
        ) in self.feature_params:
            features_compute[feature_name] = self.calc_bp_feature(
                bp_feature, feature_name, data[ch_idx, f_band_idx, -seglen:]
            )

        return features_compute

    def calc_bp_feature(self, bp_feature, feature_name, data):
        match bp_feature:
            case "activity":
                feature_calc = np.var(data)
                if self.bp_settings.log_transform:
                    feature_calc = np.log10(feature_calc)
                if self.KF_dict:
                    feature_calc = self.update_KF(feature_calc, feature_name)
            case "mobility":
                feature_calc = np.sqrt(np.var(np.diff(data)) / np.var(data))
            case "complexity":
                feature_calc = self.calc_complexity(data)
            case _:
                raise ValueError(f"Unknown bandpower feature: {bp_feature}")

        return np.nan_to_num(feature_calc)

    @staticmethod
    def calc_complexity(data: np.ndarray) -> float:
        dat_deriv = np.diff(data)
        deriv_variance = np.var(dat_deriv)
        mobility = np.sqrt(deriv_variance / np.var(data))
        dat_deriv_2_var = np.var(np.diff(dat_deriv))
        deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)

        return deriv_mobility / mobility
