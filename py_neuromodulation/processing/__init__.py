from .artifacts import PARRMArtifactRejection
from .data_preprocessor import DataPreprocessor
from .projection import Projection
from .normalization import FeatureNormalizer, RawNormalizer
from .resample import Resampler
from .rereference import ReReferencer

# Expose Notch filter also in the processing module, as it is used as a data preprocessing step
from py_neuromodulation.filter import NotchFilter

__all__ = [
    "PARRMArtifactRejection",
    "DataPreprocessor",
    "Projection",
    "FeatureNormalizer",
    "RawNormalizer",
    "Resampler",
    "ReReferencer",
    "NotchFilter",
]
