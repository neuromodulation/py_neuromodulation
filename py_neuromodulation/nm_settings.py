"""Module for handling settings."""

from pathlib import PurePath, Path
from pydantic import Field, model_validator

from py_neuromodulation import PYNM_DIR, logger
from py_neuromodulation.nm_types import (
    FeatureSelector,
    FrequencyRange,
    PreprocessorName,
    _PathLike,
)

from py_neuromodulation.nm_types import NMBaseModel
from py_neuromodulation.nm_filter_preprocessing import FilterSettings
from py_neuromodulation.nm_kalmanfilter import KalmanSettings
from py_neuromodulation.nm_projection import ProjectionSettings
from py_neuromodulation.nm_bispectra import BispectraSettings
from py_neuromodulation.nm_nolds import NoldsSettings
from py_neuromodulation.nm_mne_connectivity import MNEConnectivitySettings
from py_neuromodulation.nm_fooof import FooofSettings
from py_neuromodulation.nm_coherence import CoherenceSettings
from py_neuromodulation.nm_sharpwaves import SharpwaveSettings
from py_neuromodulation.nm_oscillatory import OscillatorySettings, BandpassSettings
from py_neuromodulation.nm_bursts import BurstSettings
from py_neuromodulation.nm_normalization import NormMethod, NormalizationSettings
from py_neuromodulation.nm_resample import ResamplerSettings


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


class PostprocessingSettings(FeatureSelector):
    feature_normalization: bool = True
    project_cortex: bool = False
    project_subcortex: bool = False


class NMSettings(NMBaseModel):
    # General settings
    sampling_rate_features_hz: float = Field(default=10, gt=0)
    segment_length_features_ms: float = Field(default=1000, gt=0)
    frequency_ranges_hz: dict[str, FrequencyRange] = {
        "theta": FrequencyRange(4, 8),
        "alpha": FrequencyRange(8, 12),
        "low beta": FrequencyRange(13, 20),
        "high beta": FrequencyRange(20, 35),
        "low gamma": FrequencyRange(60, 80),
        "high gamma": FrequencyRange(90, 200),
        "HFA": FrequencyRange(200, 400)
    }

    # Preproceessing settings
    raw_resampling_settings: ResamplerSettings = ResamplerSettings()
    preprocessing_filter: FilterSettings = FilterSettings()
    raw_normalization_settings: NormalizationSettings = NormalizationSettings()
    feature_normalization_settings: NormalizationSettings = NormalizationSettings()

    # Postprocessing settings
    preprocessing: list[PreprocessorName] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]
    postprocessing: PostprocessingSettings = PostprocessingSettings()
    project_cortex_settings: ProjectionSettings = ProjectionSettings(max_dist_mm=20)
    project_subcortex_settings: ProjectionSettings = ProjectionSettings(max_dist_mm=5)

    # Feature settings
    # Maybe this should be a subclass of FeatureSelector
    features: Features = Features()

    fft_settings: OscillatorySettings = OscillatorySettings()
    welch_settings: OscillatorySettings = OscillatorySettings()
    stft_settings: OscillatorySettings = OscillatorySettings()
    bandpass_filter_settings: BandpassSettings = BandpassSettings()
    kalman_filter_settings: KalmanSettings = KalmanSettings()
    burst_settings: BurstSettings = BurstSettings()
    sharpwave_analysis_settings: SharpwaveSettings = SharpwaveSettings()
    mne_connectivity: MNEConnectivitySettings = MNEConnectivitySettings()
    coherence: CoherenceSettings = CoherenceSettings()
    fooof: FooofSettings = FooofSettings()
    nolds_features: NoldsSettings = NoldsSettings()
    bispectrum: BispectraSettings = BispectraSettings()

    @model_validator(mode="after")
    def validate_settings(self):

        if len(self.features.get_enabled()) == 0:
            raise ValueError("At least one feature must be selected.")

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
                              (e.g. /path/to/exp/exp_prefix[_SETTINGS.json])

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

                with open(path) as f:
                    model_dict = yaml.safe_load(f)
            case _:
                raise ValueError("File format not supported.")

        return NMSettings(**model_dict)

    @staticmethod
    def get_default() -> "NMSettings":
        return NMSettings.from_file(PYNM_DIR / "nm_settings.json")

    @staticmethod
    def list_normalization_methods() -> list[NormMethod]:
        return NormalizationSettings.list_normalization_methods()

    def save(
        self, path_out: _PathLike, folder_name: str = "", format: str = "json"
    ) -> None:
        path_out = PurePath(path_out)
        filename = f"SETTINGS.{format}"

        if folder_name:
            path_out = path_out / folder_name
            filename = f"{folder_name}_{filename}"

        path_out = path_out / filename

        with open(path_out, "w") as f:
            match format:
                case "json":
                    f.write(self.model_dump_json(indent=4))
                case "yaml":
                    import yaml

                    yaml.dump(self.model_dump(), f, default_flow_style=False)

        logger.info(f"Settings saved to {path_out}")


# For retrocompatibility with previous versions of PyNM
def get_default_settings() -> NMSettings:
    return NMSettings.get_default()


def reset_settings(settings: NMSettings) -> NMSettings:
    return settings.reset()


def set_settings_fast_compute() -> NMSettings:
    return NMSettings.get_fast_compute()


def test_settings(settings: NMSettings) -> NMSettings:
    return settings.validate()
