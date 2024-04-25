from collections.abc import Iterable, Callable
import numpy as np
from pybispectra import compute_fft, WaveShape

from py_neuromodulation.nm_features_abc import Feature


class Bispectra(Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: int | float
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.s = settings
        self.f1s = settings["bispectrum"]["f1s"]
        self.f2s = settings["bispectrum"]["f2s"]

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        s = settings

        def test_range(f_name, filter_range):
            assert isinstance(
                filter_range[0],
                int,
            ), f"bispectrum frequency range {f_name} needs to be of type int, got {filter_range[0]}"
            assert isinstance(
                filter_range[1],
                int,
            ), f"bispectrum frequency range {f_name} needs to be of type int, got {filter_range[1]}"
            assert (
                filter_range[1] > filter_range[0]
            ), f"second frequency range value needs to be higher than first one, got {filter_range}"
            assert filter_range[0] < sfreq and filter_range[1] < sfreq, (
                "filter frequency range has to be smaller than sfreq, "
                f"got sfreq {sfreq} and filter range {filter_range}"
            )

        test_range("f1s", s["bispectrum"]["f1s"])
        test_range("f2s", s["bispectrum"]["f2s"])

        for feature_name, val in s["bispectrum"]["components"].items():
            assert isinstance(
                val, bool
            ), f"bispectrum component {feature_name} has to be of type bool, got {val}"

        for feature_name, val in s["bispectrum"]["bispectrum_features"].items():
            assert isinstance(
                val, bool
            ), f"bispectrum feature {feature_name} has to be of type bool, got {val}"

        assert (
            f_band_bispectrum in s["frequency_ranges_hz"]
            for f_band_bispectrum in s["bispectrum"]["frequency_bands"]
        ), (
            "bispectrum selected frequency bands don't match the ones"
            "specified in s['frequency_ranges_hz']"
            f"bispectrum frequency bands: {s['bispectrum']['frequency_bands']}"
            f"specified frequency_ranges_hz: {s['frequency_ranges_hz']}"
        )

    def compute_bs_features(
        self,
        spectrum_ch: np.ndarray,
        features_compute: dict,
        ch_name: str,
        component: str,
        f_band: str | None,
    ) -> dict:
        func : Callable
        for bispectrum_feature in self.s["bispectrum"]["bispectrum_features"]:
            if bispectrum_feature == "mean":
                func  = np.nanmean
            if bispectrum_feature == "sum":
                func = np.nansum
            if bispectrum_feature == "var":
                func = np.nanvar

            if f_band is not None:
                str_feature = "_".join(
                    [
                        ch_name,
                        "Bispectrum",
                        component,
                        bispectrum_feature,
                        f_band,
                    ]
                )
            else:
                str_feature = "_".join(
                    [
                        ch_name,
                        "Bispectrum",
                        component,
                        bispectrum_feature,
                        "whole_fband_range",
                    ]
                )

            features_compute[str_feature] = func(spectrum_ch)

        return features_compute

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            fft_coeffs, freqs = compute_fft(
                data=np.expand_dims(data[ch_idx, :], axis=(0, 1)),
                sampling_freq=self.sfreq,
                n_points=data.shape[1],
                verbose=False,
            )

            f_spectrum_range = freqs[
                np.logical_and(
                    freqs >= np.min([self.f1s, self.f2s]),
                    freqs <= np.max([self.f1s, self.f2s]),
                )
            ]

            waveshape = WaveShape(
                data=fft_coeffs,
                freqs=freqs,
                sampling_freq=self.sfreq,
                verbose=False,
            )

            waveshape.compute(
                f1s=(self.f1s[0], self.f1s[-1]), f2s=(self.f2s[0], self.f2s[-1])
            )

            bispectrum = np.squeeze(waveshape.results._data)

            for component in self.s["bispectrum"]["components"]:
                if self.s["bispectrum"]["components"][component]:
                    if component == "real":
                        spectrum_ch = bispectrum.real
                    if component == "imag":
                        spectrum_ch = bispectrum.imag
                    if component == "absolute":
                        spectrum_ch = np.abs(bispectrum)
                    if component == "phase":
                        spectrum_ch = np.angle(bispectrum)

                for fb in self.s["bispectrum"]["frequency_bands"]:
                    range_ = (
                        f_spectrum_range >= self.s["frequency_ranges_hz"][fb][0]
                    ) & (
                        f_spectrum_range <= self.s["frequency_ranges_hz"][fb][1]
                    )
                    # waveshape.results.plot()
                    data_bs = spectrum_ch[range_, range_]

                    features_compute = self.compute_bs_features(
                        data_bs, features_compute, ch_name, component, fb
                    )

                if self.s["bispectrum"][
                    "compute_features_for_whole_fband_range"
                ]:
                    features_compute = self.compute_bs_features(
                        spectrum_ch, features_compute, ch_name, component, None
                    )

        return features_compute
