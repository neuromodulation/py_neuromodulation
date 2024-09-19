from .artifacts import PARRMArtifactRejection
from .data_preprocessor import DataPreprocessor
from .projection import Projection, ProjectionSettings
from .normalization import FeatureNormalizer, RawNormalizer, NormalizationSettings
from .resample import Resampler, ResamplerSettings
from .rereference import ReReferencer
from .filter_preprocessing import PreprocessingFilter, FilterSettings

# Expose Notch filter also in the processing module, as it is used as a data preprocessing step
from py_neuromodulation.filter import NotchFilter

__all__ = [
    "PARRMArtifactRejection",
    "DataPreprocessor",
    "Projection",
    "ProjectionSettings",
    "FeatureNormalizer",
    "RawNormalizer",
    "NormalizationSettings",
    "Resampler",
    "ResamplerSettings",
    "ReReferencer",
    "NotchFilter",
    "PreprocessingFilter",
    "FilterSettings",
]
