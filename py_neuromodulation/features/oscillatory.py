from collections.abc import Sequence
import numpy as np
from itertools import product

from py_neuromodulation.utils.types import NMBaseModel, BoolSelector, NMFeature
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py_neuromodulation.stream.settings import NMSettings


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
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: int
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
        ch_names: Sequence[str],
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

    def calc_feature(self, data: np.ndarray) -> dict:
        data = data[:, self.window_samples :]

        from scipy.fft import rfft

        Z = np.abs(rfft(data))  # type: ignore

        if self.settings.log_transform:
            Z = np.log10(Z)

        feature_results = {}

        for f_band_name, idx_range in self.idx_range:
            # TODO Can we get rid of this for-loop? Hard to vectorize windows of different lengths...
            Z_band = Z[:, idx_range]  # Data for all channels

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=1)

                for ch_idx, ch_name in enumerate(self.ch_names):
                    feature_results[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                feature_results[f"{ch_name}_fft_psd_{int(f)}"] = Z[ch_idx][idx]

        return feature_results


class Welch(OscillatoryFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Sequence[str],
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

    def calc_feature(self, data: np.ndarray) -> dict:
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

        feature_results = {}

        for f_band_name, idx_range in self.idx_range:
            Z_band = Z[:, idx_range]

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=1)

                for ch_idx, ch_name in enumerate(self.ch_names):
                    feature_results[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                feature_results[f"{ch_name}_welch_psd_{str(f)}"] = Z[ch_idx][idx]

        return feature_results


class STFT(OscillatoryFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Sequence[str],
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

    def calc_feature(self, data: np.ndarray) -> dict:
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

        feature_results = {}

        for f_band_name, idx_range in self.idx_range:
            Z_band = Z[:, idx_range, :]

            for est_name, est_fun in self.estimators:
                result = est_fun(Z_band, axis=(1, 2))

                for ch_idx, ch_name in enumerate(self.ch_names):
                    feature_results[
                        f"{ch_name}_{self.osc_feature_name}_{f_band_name}_{est_name}"
                    ] = result[ch_idx]

        if self.settings.return_spectrum:
            combinations = product(enumerate(self.ch_names), enumerate(self.freqs))
            for (ch_idx, ch_name), (idx, f) in combinations:
                feature_results[f"{ch_name}_stft_psd_{str(f)}"] = Z[ch_idx].mean(
                    axis=1
                )[idx]

        return feature_results
