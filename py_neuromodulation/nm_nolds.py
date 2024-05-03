import numpy as np
from collections.abc import Iterable
import nolds

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_oscillatory import BandPower
from py_neuromodulation import logger


class Nolds(NMFeature):
    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.s = settings
        self.ch_names = ch_names

        if len(self.s["nolds_features"]["data"]["frequency_bands"]) > 0:
            self.bp_filter = BandPower(settings, ch_names, sfreq, use_kf=False)

    @staticmethod
    def test_settings(
        s: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ):
        nolds_feature_cols = [
            "sample_entropy",
            "correlation_dimension",
            "lyapunov_exponent",
            "hurst_exponent",
            "detrended_fluctutaion_analysis",
        ]
        if sum([s["nolds_features"][f] for f in nolds_feature_cols]) == 0:
            logger.warn("nolds feature enabled, but no nolds_feature type selected")

        for fb in s["nolds_features"]["data"]["frequency_bands"]:
            assert (
                fb in list(s["frequency_ranges_hz"].keys())
            ), f"{fb} selected in nolds_features, but not defined in s['frequency_ranges_hz']"

    def calc_feature(
        self,
        data: np.ndarray,
        features_compute: dict,
    ) -> dict:
        data = np.nan_to_num(data)
        if self.s["nolds_features"]["data"]["raw"]:
            features_compute = self.calc_nolds(data, features_compute)
        if len(self.s["nolds_features"]["data"]["frequency_bands"]) > 0:
            data_filt = self.bp_filter.bandpass_filter.filter_data(data)

            for f_band_idx, f_band in enumerate(
                self.s["nolds_features"]["data"]["frequency_bands"]
            ):
                # filter data now for a specific fband and pass to calc_nolds
                features_compute = self.calc_nolds(
                    data_filt[:, f_band_idx, :], features_compute, f_band
                )  # ch, bands, samples
        return features_compute

    def calc_nolds(
        self, data: np.ndarray, features_compute: dict, data_str: str = "raw"
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            dat = data[ch_idx, :]
            empty_arr = dat.sum() == 0
            if self.s["nolds_features"]["sample_entropy"]:
                features_compute[f"{ch_name}_nolds_sample_entropy"] = (
                    nolds.sampen(dat) if not empty_arr else 0
                )
            if self.s["nolds_features"]["correlation_dimension"]:
                features_compute[
                    f"{ch_name}_nolds_correlation_dimension_{data_str}"
                ] = nolds.corr_dim(dat, emb_dim=2) if not empty_arr else 0
            if self.s["nolds_features"]["lyapunov_exponent"]:
                features_compute[f"{ch_name}_nolds_lyapunov_exponent_{data_str}"] = (
                    nolds.lyap_r(dat) if not empty_arr else 0
                )
            if self.s["nolds_features"]["hurst_exponent"]:
                features_compute[f"{ch_name}_nolds_hurst_exponent_{data_str}"] = (
                    nolds.hurst_rs(dat) if not empty_arr else 0
                )
            if self.s["nolds_features"]["detrended_fluctutaion_analysis"]:
                features_compute[
                    f"{ch_name}_nolds_detrended_fluctutaion_analysis_{data_str}"
                ] = nolds.dfa(dat) if not empty_arr else 0

        return features_compute
