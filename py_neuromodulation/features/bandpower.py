import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING
from pydantic import field_validator

from py_neuromodulation.utils.types import NMBaseModel, BoolSelector, NMFeature

if TYPE_CHECKING:
    from py_neuromodulation.stream.settings import NMSettings
    from py_neuromodulation.filter import KalmanSettings


class BandpowerFeatures(BoolSelector):
    activity: bool = True
    mobility: bool = False
    complexity: bool = False


class BandPowerSettings(NMBaseModel):
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
        ch_names: Sequence[str],
        sfreq: float,
        use_kf: bool | None = None,
    ) -> None:
        settings.validate()

        self.bp_settings: BandPowerSettings = settings.bandpass_filter_settings
        self.kalman_filter_settings: KalmanSettings = settings.kalman_filter_settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict: dict = {}

        from py_neuromodulation.filter import MNEFilter

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
        from py_neuromodulation.filter import define_KF

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

    def calc_feature(self, data: np.ndarray) -> dict:
        data = self.bandpass_filter.filter_data(data)

        feature_results = {}
        for (
            ch_idx,
            f_band_idx,
            seglen,
            bp_feature,
            feature_name,
        ) in self.feature_params:
            feature_results[feature_name] = self.calc_bp_feature(
                bp_feature, feature_name, data[ch_idx, f_band_idx, -seglen:]
            )

        return feature_results

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
