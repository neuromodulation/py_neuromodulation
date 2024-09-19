"""Module for handling settings."""

from pathlib import Path
from typing import ClassVar
from pydantic import Field, model_validator

from py_neuromodulation import PYNM_DIR, logger, user_features

from py_neuromodulation.utils.types import (
    BoolSelector,
    FrequencyRange,
    PreprocessorName,
    _PathLike,
    NMBaseModel,
    NormMethod,
)

from py_neuromodulation.processing.filter_preprocessing import FilterSettings
from py_neuromodulation.processing.normalization import NormalizationSettings
from py_neuromodulation.processing.resample import ResamplerSettings
from py_neuromodulation.processing.projection import ProjectionSettings

from py_neuromodulation.filter import KalmanSettings
from py_neuromodulation.features import BispectraSettings
from py_neuromodulation.features import NoldsSettings
from py_neuromodulation.features import MNEConnectivitySettings
from py_neuromodulation.features import FooofSettings
from py_neuromodulation.features import CoherenceSettings
from py_neuromodulation.features import SharpwaveSettings
from py_neuromodulation.features import OscillatorySettings, BandPowerSettings
from py_neuromodulation.features import BurstsSettings


class FeatureSelection(BoolSelector):
    raw_hjorth: bool = True
    return_raw: bool = True
    bandpass_filter: bool = False
    stft: bool = False
    fft: bool = True
    welch: bool = True
    sharpwave_analysis: bool = True
    fooof: bool = False
    nolds: bool = False
    coherence: bool = False
    bursts: bool = True
    linelength: bool = True
    mne_connectivity: bool = False
    bispectrum: bool = False


class PostprocessingSettings(BoolSelector):
    feature_normalization: bool = True
    project_cortex: bool = False
    project_subcortex: bool = False


class NMSettings(NMBaseModel):
    # Class variable to store instances
    _instances: ClassVar[list["NMSettings"]] = []

    # General settings
    sampling_rate_features_hz: float = Field(default=10, gt=0)
    segment_length_features_ms: float = Field(default=1000, gt=0)
    frequency_ranges_hz: dict[str, FrequencyRange] = {
        "theta": FrequencyRange(4, 8),
        "alpha": FrequencyRange(8, 12),
        "low_beta": FrequencyRange(13, 20),
        "high_beta": FrequencyRange(20, 35),
        "low_gamma": FrequencyRange(60, 80),
        "high_gamma": FrequencyRange(90, 200),
        "HFA": FrequencyRange(200, 400),
    }

    # Preproceessing settings
    preprocessing: list[PreprocessorName] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]
    raw_resampling_settings: ResamplerSettings = ResamplerSettings()
    preprocessing_filter: FilterSettings = FilterSettings()
    raw_normalization_settings: NormalizationSettings = NormalizationSettings()

    # Postprocessing settings
    postprocessing: PostprocessingSettings = PostprocessingSettings()
    feature_normalization_settings: NormalizationSettings = NormalizationSettings()
    project_cortex_settings: ProjectionSettings = ProjectionSettings(max_dist_mm=20)
    project_subcortex_settings: ProjectionSettings = ProjectionSettings(max_dist_mm=5)

    # Feature settings
    features: FeatureSelection = FeatureSelection()

    fft_settings: OscillatorySettings = OscillatorySettings()
    welch_settings: OscillatorySettings = OscillatorySettings()
    stft_settings: OscillatorySettings = OscillatorySettings()
    bandpass_filter_settings: BandPowerSettings = BandPowerSettings()
    kalman_filter_settings: KalmanSettings = KalmanSettings()
    burst_settings: BurstsSettings = BurstsSettings()
    sharpwave_analysis_settings: SharpwaveSettings = SharpwaveSettings()
    mne_connectivity_settings: MNEConnectivitySettings = MNEConnectivitySettings()
    coherence_settings: CoherenceSettings = CoherenceSettings()
    fooof_settings: FooofSettings = FooofSettings()
    nolds_settings: NoldsSettings = NoldsSettings()
    bispectrum_settings: BispectraSettings = BispectraSettings()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        for feat_name in user_features.keys():
            setattr(self.features, feat_name, True)

        NMSettings._add_instance(self)

    @classmethod
    def _add_instance(cls, instance: "NMSettings") -> None:
        """Keep track of all instances created in class variable"""
        cls._instances.append(instance)

    @classmethod
    def _add_feature(cls, feature: str) -> None:
        for instance in cls._instances:
            setattr(instance.features, feature, True)

    @classmethod
    def _remove_feature(cls, feature: str) -> None:
        for instance in cls._instances:
            delattr(instance.features, feature)

    @model_validator(mode="after")
    def validate_settings(self):
        if len(self.features.get_enabled()) == 0:
            raise ValueError("At least one feature must be selected.")

        # Replace spaces with underscores in frequency band names
        self.frequency_ranges_hz = {
            k.replace(" ", "_"): v for k, v in self.frequency_ranges_hz.items()
        }

        if self.features.bandpass_filter:
            # Check BandPass settings frequency bands
            self.bandpass_filter_settings.validate_fbands(self)

            # Check Kalman filter frequency bands
            if self.bandpass_filter_settings.kalman_filter:
                self.kalman_filter_settings.validate_fbands(self)

        for k, v in self.frequency_ranges_hz.items():
            if not isinstance(v, FrequencyRange):
                self.frequency_ranges_hz[k] = FrequencyRange.create_from(v)

        return self

    def reset(self) -> "NMSettings":
        self.features.disable_all()
        self.preprocessing = []
        self.postprocessing.disable_all()
        return self

    def set_fast_compute(self) -> "NMSettings":
        self.reset()
        self.features.fft = True
        self.preprocessing = [
            "raw_resampling",
            "notch_filter",
            "re_referencing",
        ]
        self.postprocessing.feature_normalization = True
        self.postprocessing.project_cortex = False
        self.postprocessing.project_subcortex = False

        return self

    def enable_all_features(self):
        self.features.enable_all()
        return self

    def disable_all_features(self):
        self.features.disable_all()
        return self

    @staticmethod
    def get_fast_compute() -> "NMSettings":
        return NMSettings.get_default().set_fast_compute()

    @classmethod
    def load(cls, settings: "NMSettings | _PathLike | None") -> "NMSettings":
        if isinstance(settings, cls):
            return settings.validate()
        if settings is None:
            return cls.get_default()
        return cls.from_file(str(settings))

    @staticmethod
    def from_file(PATH: _PathLike) -> "NMSettings":
        """Load settings from file or a directory.

        Args:
            PATH (_PathLike): Path to settings file or to directory containing settings file,
                              or path to experiment including experiment prefix
                              (e.g. /path/to/exp/exp_prefix[_SETTINGS.json]).
                              Supported file types are .json and .yaml

        Raises:
            ValueError: when file format is not supported.

        Returns:
            NMSettings: PyNM settings object
        """
        path = Path(PATH)

        # If directory is passed, look for settings file inside
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file() and child.suffix in [".json", ".yaml"]:
                    path = child
                    break

        # If prefix is passed, look for settings file matching prefix
        if not path.is_dir() and not path.is_file():
            for child in path.parent.iterdir():
                ext = child.suffix.lower()
                if (
                    child.is_file()
                    and ext in [".json", ".yaml"]
                    and child.name == path.stem + "_SETTINGS" + ext
                ):
                    path = child
                    break

        match path.suffix:
            case ".json":
                import json

                with open(path) as f:
                    model_dict = json.load(f)
            case ".yaml":
                import yaml

                # with open(path) as f:
                #    model_dict = yaml.safe_load(f)

                # Timon: this is potentially dangerous since python code is directly executed
                with open(path) as f:
                    model_dict = yaml.load(f, Loader=yaml.Loader)

            case _:
                raise ValueError("File format not supported.")

        return NMSettings(**model_dict)

    @staticmethod
    def get_default() -> "NMSettings":
        return NMSettings.from_file(PYNM_DIR / "default_settings.yaml")

    @staticmethod
    def list_normalization_methods() -> list[NormMethod]:
        return NormalizationSettings.list_normalization_methods()

    def save(
        self, out_dir: _PathLike = ".", prefix: str = "", format: str = "yaml"
    ) -> None:
        filename = f"{prefix}_SETTINGS.{format}" if prefix else f"SETTINGS.{format}"

        path_out = Path(out_dir) / filename

        with open(path_out, "w") as f:
            match format:
                case "json":
                    f.write(self.model_dump_json(indent=4))
                case "yaml":
                    import yaml

                    yaml.dump(self.model_dump(), f, default_flow_style=None)

        logger.info(f"Settings saved to {path_out.resolve()}")


# For retrocompatibility with previous versions of PyNM
def get_default_settings() -> NMSettings:
    return NMSettings.get_default()


def reset_settings(settings: NMSettings) -> NMSettings:
    return settings.reset()


def get_fast_compute() -> NMSettings:
    return NMSettings.get_fast_compute()


def test_settings(settings: NMSettings) -> NMSettings:
    return settings.validate()
