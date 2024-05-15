from collections.abc import Iterable
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

import numpy as np

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import FeatureSelector, FrequencyRange
from py_neuromodulation.nm_settings import NMSettings


@dataclass
class BispectraComponents(FeatureSelector):
    absolute: bool = True
    real: bool = True
    imag: bool = True
    phase: bool = True


@dataclass
class BispectraFeatures(FeatureSelector):
    mean = True
    sum = True
    var = True


@dataclass
class BispectraSettings:
    @staticmethod
    def default_bands() -> list[str]:
        return ["theta", "alpha", "low_beta", "high_beta"]

    f1s: FrequencyRange = Field(default=(5, 35), min_length=2, max_length=2)
    f2s: FrequencyRange = Field(default=(5, 35), min_length=2, max_length=2)
    compute_features_for_whole_fband_range: bool = True
    frequency_bands: list[str] = Field(default_factory=default_bands)

    components: BispectraComponents = BispectraComponents()
    bispectrum_features: BispectraFeatures = BispectraFeatures()

    @field_validator("f1s", "f2s")
    def test_range(cls, filter_range):
        assert (
            filter_range[1] > filter_range[0]
        ), f"second frequency range value needs to be higher than first one, got {filter_range}"
        return filter_range


class Bispectra(NMFeature):
    def __init__(
        self, settings: NMSettings, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.frequency_ranges_hz = settings.frequency_ranges_hz
        self.settings: BispectraSettings = settings.bispectrum

    def test_settings(self, settings: NMSettings):
        assert (
            f_band_bispectrum in settings.frequency_ranges_hz
            for f_band_bispectrum in self.settings.frequency_bands
        ), (
            "bispectrum selected frequency bands don't match the ones"
            "specified in s['frequency_ranges_hz']"
            f"bispectrum frequency bands: {self.settings.frequency_bands}"
            f"specified frequency_ranges_hz: {settings.frequency_ranges_hz}"
        )

    def compute_bs_features(
        self,
        spectrum_ch: np.ndarray,
        features_compute: dict,
        ch_name: str,
        component: str,
        f_band: str = "whole_fband_range",
    ) -> dict:
        for bispectrum_feature in self.settings.bispectrum_features.get_enabled():
            match bispectrum_feature:
                case "mean":
                    result = np.nanmean(spectrum_ch)
                case "sum":
                    result = np.nansum(spectrum_ch)
                case "var":
                    result = np.nanvar(spectrum_ch)
                case _:
                    raise ValueError(f"Unknown bispectrum feature {bispectrum_feature}")

            features_compute[
                f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_{f_band}"
            ] = result

        return features_compute

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            from pybispectra import compute_fft, WaveShape

            fft_coeffs, freqs = compute_fft(
                data=np.expand_dims(data[ch_idx, :], axis=(0, 1)),
                sampling_freq=self.sfreq,
                n_points=data.shape[1],
                verbose=False,
            )

            f_spectrum_range = freqs[
                np.logical_and(
                    freqs >= np.min([self.settings.f1s, self.settings.f2s]),
                    freqs <= np.max([self.settings.f1s, self.settings.f2s]),
                )
            ]

            waveshape = WaveShape(
                data=fft_coeffs,
                freqs=freqs,
                sampling_freq=self.sfreq,
                verbose=False,
            )

            waveshape.compute(
                f1s=(self.settings.f1s[0], self.settings.f1s[-1]),
                f2s=(self.settings.f2s[0], self.settings.f2s[-1]),
            )

            bispectrum = np.squeeze(waveshape.results._data)

            for component in self.settings.components.get_enabled():
                match component:
                    case "real":
                        spectrum_ch = bispectrum.real
                    case "imag":
                        spectrum_ch = bispectrum.imag
                    case "absolute":
                        spectrum_ch = np.abs(bispectrum)
                    case "phase":
                        spectrum_ch = np.angle(bispectrum)
                    case _:
                        raise ValueError(f"Unknown component {component}")

                for fb in self.settings.frequency_bands:
                    range_ = (f_spectrum_range >= self.frequency_ranges_hz[fb][0]) & (
                        f_spectrum_range <= self.frequency_ranges_hz[fb][1]
                    )
                    # waveshape.results.plot()
                    data_bs = spectrum_ch[range_, range_]

                    features_compute = self.compute_bs_features(
                        data_bs, features_compute, ch_name, component, fb
                    )

                if self.settings.compute_features_for_whole_fband_range:
                    features_compute = self.compute_bs_features(
                        spectrum_ch,
                        features_compute,
                        ch_name,
                        component,
                        "whole_fband_range",
                    )

        return features_compute
