from collections.abc import Iterable
import numpy as np
from fooof import FOOOF
from scipy import fft

from pydantic.dataclasses import dataclass
from nm_types import FrequencyRange
from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_settings import NMSettings
from py_neuromodulation.nm_types import FeatureSelector
from py_neuromodulation import logger


@dataclass
class FooofAperiodicSettings(FeatureSelector):
    exponent: bool = True
    offset: bool = True
    knee: bool = True


@dataclass
class FooofPeriodicSettings(FeatureSelector):
    center_frequency: bool = False
    band_width: bool = False
    height_over_ap: bool = False


@dataclass
class FooofSettings:
    aperiodic: FooofAperiodicSettings
    periodic: FooofPeriodicSettings
    windowlength_ms: float = 800
    peak_width_limits: FrequencyRange = FrequencyRange(0.5, 12)
    max_n_peaks: int = 3
    min_peak_height: float = 0
    peak_threshold: float = 2
    freq_range_hz: FrequencyRange = FrequencyRange(2, 40)
    knee: bool = True


class FooofAnalyzer(NMFeature):
    feat_name_map = {
        "exponent": "exp",
        "offset": "offset",
        "knee": "knee_frequency",
        "center_frequency": "cf",
        "band_width": "bw",
        "height_over_ap": "pw",
    }

    def __init__(
        self, settings: NMSettings, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.settings = settings.fooof
        self.sfreq = sfreq
        self.ch_names = ch_names

        self.ap_mode = "knee" if self.settings.knee else "fixed"

        self.num_samples = int(self.settings.windowlength_ms * sfreq / 1000)

        self.f_vec = np.arange(0, int(self.num_samples / 2) + 1, 1)

        assert (
            settings.fooof.windowlength_ms <= settings.segment_length_features_ms
        ), f"fooof windowlength_ms ({settings.fooof.windowlength_ms}) needs to be smaller equal than segment_length_features_ms ({settings.segment_length_features_ms})."

        assert (
            settings.fooof.freq_range_hz[0] < sfreq
            and settings.fooof.freq_range_hz[1] < sfreq
        ), f"fooof frequency range needs to be below sfreq, got {settings.fooof.freq_range_hz}"

    def _get_spectrum(self, data: np.ndarray):
        """return absolute value fft spectrum"""

        data = data[-self.num_samples :]
        Z = np.abs(fft.rfft(data))

        return Z

    def calc_feature(
        self,
        data: np.ndarray,
        features_compute: dict,
    ) -> dict:
        for ch_idx, ch_name in enumerate(self.ch_names):
            spectrum = self._get_spectrum(data[ch_idx, :])

            try:
                fm = FOOOF(
                    aperiodic_mode=self.ap_mode,
                    peak_width_limits=self.settings.peak_width_limits,
                    max_n_peaks=self.settings.max_n_peaks,
                    min_peak_height=self.settings.min_peak_height,
                    peak_threshold=self.settings.peak_threshold,
                    verbose=False,
                )
                fm.fit(self.f_vec, spectrum, self.settings.freq_range_hz)
            except Exception as e:
                logger.critical(e, exc_info=True)
                raise e

            FIT_PASSED = fm.fooofed_spectrum_ is not None

            for feat in self.settings.aperiodic.get_enabled():
                f_name = f"{ch_name}_fooof_a_{self.feat_name_map[feat]}"

                if not FIT_PASSED:
                    features_compute[f_name] = None

                elif (
                    feat == "knee"
                    and fm.get_params("aperiodic_params", "exponent") == 0
                ):
                    features_compute[f_name] = None

                else:
                    params = fm.get_params("aperiodic_params", feat)

                    if feat == "knee":
                        params = params ** (
                            1 / fm.get_params("aperiodic_params", "exponent")
                        )

                    features_compute[f_name] = np.nan_to_num(params)

            peaks_dict: dict[str, np.ndarray | None] = {
                "bw": fm.get_params("peak_params", "BW") if FIT_PASSED else None,
                "cf": fm.get_params("peak_params", "CF") if FIT_PASSED else None,
                "pw": fm.get_params("peak_params", "PW") if FIT_PASSED else None,
            }

            if type(peaks_dict["bw"]) is np.float64 or peaks_dict["bw"] is None:
                peaks_dict["bw"] = [peaks_dict["bw"]]
                peaks_dict["cf"] = [peaks_dict["cf"]]
                peaks_dict["pw"] = [peaks_dict["pw"]]

            for peak_idx in range(self.settings.max_n_peaks):
                for feat in self.settings.periodic.get_enabled():
                    f_name = f"{ch_name}_fooof_p_{peak_idx}_{self.feat_name_map[feat]}"

                    features_compute[f_name] = (
                        peaks_dict[self.feat_name_map[feat]][peak_idx]
                        if peak_idx < len(peaks_dict[self.feat_name_map[feat]])
                        else None
                    )

        return features_compute
