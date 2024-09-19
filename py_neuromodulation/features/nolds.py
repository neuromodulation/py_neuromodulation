import numpy as np
from collections.abc import Iterable

from typing import TYPE_CHECKING

from py_neuromodulation.utils.types import NMFeature, BoolSelector, NMBaseModel

from pydantic import field_validator

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class NoldsFeatures(BoolSelector):
    sample_entropy: bool = False
    correlation_dimension: bool = False
    lyapunov_exponent: bool = True
    hurst_exponent: bool = False
    detrended_fluctuation_analysis: bool = False


class NoldsSettings(NMBaseModel):
    raw: bool = True
    frequency_bands: list[str] = ["low_beta"]
    features: NoldsFeatures = NoldsFeatures()

    @field_validator("frequency_bands")
    def fbands_spaces_to_underscores(cls, frequency_bands):
        return [f.replace(" ", "_") for f in frequency_bands]


class Nolds(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.settings = settings.nolds_features
        self.ch_names = ch_names

        if len(self.settings.frequency_bands) > 0:
            from py_neuromodulation.oscillatory import BandPower

            self.bp_filter = BandPower(settings, ch_names, sfreq, use_kf=False)

        # Check if the selected frequency bands are defined in the global settings
        for fb in settings.nolds_features.frequency_bands:
            assert (
                fb in settings.frequency_ranges_hz
            ), f"{fb} selected in nolds_features, but not defined in s['frequency_ranges_hz']"

    def calc_feature(self, data: np.ndarray) -> dict:
        feature_results = {}
        data = np.nan_to_num(data)
        if self.settings.raw:
            feature_results = self.calc_nolds(data, feature_results)
        if len(self.settings.frequency_bands) > 0:
            data_filt = self.bp_filter.bandpass_filter.filter_data(data)

            for f_band_idx, f_band in enumerate(self.settings.frequency_bands):
                # filter data now for a specific fband and pass to calc_nolds
                feature_results = self.calc_nolds(
                    data_filt[:, f_band_idx, :], feature_results, f_band
                )  # ch, bands, samples
        return feature_results

    def calc_nolds(
        self, data: np.ndarray, feature_results: dict, data_str: str = "raw"
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            for f_name in self.settings.features.get_enabled():
                feature_results[f"{ch_name}_nolds_{f_name}_{data_str}"] = (
                    self.calc_nolds_feature(f_name, data[ch_idx, :])
                    if data[ch_idx, :].sum()
                    else 0
                )

        return feature_results

    @staticmethod
    def calc_nolds_feature(f_name: str, dat: np.ndarray):
        import nolds

        match f_name:
            case "sample_entropy":
                return nolds.sampen(dat)
            case "correlation_dimension":
                return nolds.corr_dim(dat, emb_dim=2)
            case "lyapunov_exponent":
                return nolds.lyap_r(dat)
            case "hurst_exponent":
                return nolds.hurst_rs(dat)
            case "detrended_fluctuation_analysis":
                return nolds.dfa(dat)
            case _:
                raise ValueError(f"Invalid nolds feature name: {f_name}")
