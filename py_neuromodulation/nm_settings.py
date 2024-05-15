"""Module for handling settings."""

from pydantic.dataclasses import dataclass
from pydantic import Field, BaseModel, model_validator
from typing import Iterable, TYPE_CHECKING

from py_neuromodulation.nm_filter_preprocessing import FilterSettings
from py_neuromodulation.nm_types import (
    FeatureSelector,
    FrequencyRange,
    FeatureName,
    PreprocessorName,
)

if TYPE_CHECKING:
    from py_neuromodulation.nm_kalmanfilter import KalmanSettings
    from py_neuromodulation.nm_types import FrequencyRange
    from py_neuromodulation.nm_projection import ProjectionSettings
    from py_neuromodulation.nm_bispectra import BispectraSettings
    from py_neuromodulation.nm_nolds import NoldsSettings
    from py_neuromodulation.nm_mne_connectivity import MNEConnectivitySettings
    from py_neuromodulation.nm_fooof import FooofSettings
    from py_neuromodulation.nm_coherence import CoherenceSettings
    from py_neuromodulation.nm_sharpwaves import SharpwaveSettings
    from py_neuromodulation.nm_oscillatory import OscillatorySettings, BandpassSettings
    from py_neuromodulation.nm_bursts import BurstSettings
    from py_neuromodulation.nm_normalization import NormalizationSettings


def get_invalid_keys(
    input_seq: Iterable[str],
    validation_dict: dict,
) -> list[str]:
    return [v for v in input_seq if v not in validation_dict]


@dataclass
class ProcessorSettings:
    pass


@dataclass
class Features(FeatureSelector):
    raw_hjorth: bool = True
    return_raw: bool = True
    bandpass_filter: bool = False
    stft: bool = False
    fft: bool = True
    welch: bool = True
    sharpwave_analysis: bool = True
    fooof: bool = False
    nolds: bool = True
    coherence: bool = True
    bursts: bool = False
    linelength: bool = False
    mne_connectivity: bool = False
    bispectrum: bool = False


@dataclass
class PostprocessingSettings(FeatureSelector):
    feature_normalization: bool = True
    project_cortex: bool = False
    project_subcortex: bool = False


class NMSettings(BaseModel):
    # General settings
    sampling_rate_features_hz: float = Field(default=10, gt=0, alias="sfreq")
    segment_length_features_ms: float = Field(default=1000, gt=0)
    frequency_ranges_hz: dict[str, FrequencyRange]

    # Preproceessing settings
    # raw_resampling_settings: # TONI: is this ever used?
    preprocessing_filter: FilterSettings
    raw_normalization_settings: NormalizationSettings
    feature_normalization_settings: NormalizationSettings

    # Postprocessing settings
    preprocessing: list[PreprocessorName] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]
    postprocessing: PostprocessingSettings
    project_cortex_settings: ProjectionSettings
    project_subcortex_settings: ProjectionSettings

    # Feature settings
    # Maybe this should be a subclass of FeatureSelector
    features: dict[FeatureName, bool] = {
        "raw_hjorth": True,
        "return_raw": True,
        "bandpass_filter": False,
        "stft": False,
        "fft": True,
        "welch": True,
        "sharpwave_analysis": True,
        "fooof": False,
        "bursts": True,
        "linelength": True,
        "coherence": False,
        "nolds": False,
        "mne_connectivity": False,
        "bispectrum": False,
    }
    
    fft_settings: OscillatorySettings
    welch_settings: OscillatorySettings
    stft_settings: OscillatorySettings
    bandpass_filter_settings: BandpassSettings
    kalman_filter_settings: KalmanSettings
    burst_settings: BurstSettings
    sharpwave_analysis_settings: SharpwaveSettings
    mne_connectivity: MNEConnectivitySettings
    coherence: CoherenceSettings
    fooof: FooofSettings
    nolds_features: NoldsSettings
    bispectrum: BispectraSettings

    # Validation
    @model_validator(mode="after")
    def validate_settings(self):
        # Check Kalman filter frequency bands
        assert all(
            [
                item in self.frequency_ranges_hz
                for item in self.kalman_filter_settings.frequency_bands
            ]
        ), (
            "Frequency bands for Kalman filter must also be specified in "
            "bandpass_filter_settings."
        )

        if not any(self.features.values()):
            raise ValueError("At least one feature must be selected.")

        return self

    def reset(self) -> None:
        self.features = {k: False for k in self.features}
        self.preprocessing = []

    def set_fast_compute(self) -> None:
        self.reset()
        self.features["fft"] = True
        self.preprocessing = [
            "raw_resampling",
            "notch_filter",
            "re_referencing",
        ]
        self.postprocessing.feature_normalization = True
        self.postprocessing.project_cortex = False
        self.postprocessing.project_subcortex = False


# def get_default_settings() -> NMSettings:
#     return NMSettings()
