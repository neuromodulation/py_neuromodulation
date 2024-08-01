from collections.abc import Iterable
from pydantic import field_validator
from py_neuromodulation.nm_types import NMBaseModel
from typing import TYPE_CHECKING, Callable

import numpy as np

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import BoolSelector, FrequencyRange

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


class BispectraComponents(BoolSelector):
    absolute: bool = True
    real: bool = True
    imag: bool = True
    phase: bool = True


class BispectraFeatures(BoolSelector):
    mean: bool = True
    sum: bool = True
    var: bool = True


class BispectraSettings(NMBaseModel):
    f1s: FrequencyRange = FrequencyRange(5, 35)
    f2s: FrequencyRange = FrequencyRange(5, 35)
    compute_features_for_whole_fband_range: bool = True
    frequency_bands: list[str] = ["theta", "alpha", "low_beta", "high_beta"]

    components: BispectraComponents = BispectraComponents()
    bispectrum_features: BispectraFeatures = BispectraFeatures()

    @field_validator("f1s", "f2s")
    def test_range(cls, filter_range):
        assert (
            filter_range[1] > filter_range[0]
        ), f"second frequency range value needs to be higher than first one, got {filter_range}"
        return filter_range

    @field_validator("frequency_bands")
    def fbands_spaces_to_underscores(cls, frequency_bands):
        return [f.replace(" ", "_") for f in frequency_bands]


FEATURE_DICT: dict[str, Callable] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "var": np.nanvar,
}

COMPONENT_DICT: dict[str, Callable] = {
    "real": lambda obj: getattr(obj, "real"),
    "imag": lambda obj: getattr(obj, "imag"),
    "absolute": np.abs,
    "phase": np.angle,
}


class Bispectra(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.frequency_ranges_hz = settings.frequency_ranges_hz
        self.settings: BispectraSettings = settings.bispectrum

        assert all(
            f_band_bispectrum in settings.frequency_ranges_hz
            for f_band_bispectrum in self.settings.frequency_bands
        ), (
            "bispectrum selected frequency bands don't match the ones"
            "specified in s['frequency_ranges_hz']"
            f"bispectrum frequency bands: {self.settings.frequency_bands}"
            f"specified frequency_ranges_hz: {settings.frequency_ranges_hz}"
        )

        self.used_features = self.settings.bispectrum_features.get_enabled()

        self.min_freq = min(
            self.settings.f1s.frequency_low_hz, self.settings.f2s.frequency_low_hz
        )
        self.max_freq = max(
            self.settings.f1s.frequency_high_hz, self.settings.f2s.frequency_high_hz
        )

        # self.freqs: np.ndarray = np.array([]) # In case we pre-computed this

    def calc_feature(self, data: np.ndarray) -> dict:
        from pybispectra import compute_fft, WaveShape

        # PyBispectra's compute_fft uses PQDM to parallelize the calculation per channel
        # Is this necessary? Maybe the overhead of parallelization is not worth it
        # considering that we incur in it once per batch of data
        fft_coeffs, freqs = compute_fft(
            data=np.expand_dims(data, axis=(0)),
            sampling_freq=self.sfreq,
            n_points=data.shape[1],
            verbose=False,
        )

        # freqs is batch independent, except for the last batch perhaps (if it has different shape)
        # but it's computed by compute_fft regardless so no advantage in pre-computing it
        # if not self.freqs = self.freqs = np.fft.rfftfreq(n=data.shape[1], d = 1 / sfreq)

        # fft_coeffs shape: [epochs, channels, frequencies]

        f_spectrum_range = freqs[
            np.logical_and(freqs >= self.min_freq, freqs <= self.max_freq)
        ]

        waveshape = WaveShape(
            data=fft_coeffs,
            freqs=freqs,
            sampling_freq=self.sfreq,
            verbose=False,
        )

        waveshape.compute(
            f1s=tuple(self.settings.f1s),  # type: ignore
            f2s=tuple(self.settings.f2s),  # type: ignore
        )

        feature_results = {}
        for ch_idx, ch_name in enumerate(self.ch_names):
            bispectrum = waveshape._bicoherence[
                ch_idx
            ]  # Same as waveshape.results._data, skips a copy

            for component in self.settings.components.get_enabled():
                spectrum_ch = COMPONENT_DICT[component](bispectrum)

                for fb in self.settings.frequency_bands:
                    range_ = (f_spectrum_range >= self.frequency_ranges_hz[fb][0]) & (
                        f_spectrum_range <= self.frequency_ranges_hz[fb][1]
                    )
                    # waveshape.results.plot()
                    data_bs = spectrum_ch[range_, range_]

                    for bispectrum_feature in self.used_features:
                        feature_results[
                            f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_{fb}"
                        ] = FEATURE_DICT[bispectrum_feature](data_bs)

                        if self.settings.compute_features_for_whole_fband_range:
                            feature_results[
                                f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_whole_fband_range"
                            ] = FEATURE_DICT[bispectrum_feature](spectrum_ch)

        return feature_results
