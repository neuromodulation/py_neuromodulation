from .artifacts import PARRMArtifactRejection as PARRMArtifactRejection
from .data_preprocessor import DataPreprocessor as DataPreprocessor
from .projection import Projection as Projection
from .normalization import (
    FeatureNormalizer as FeatureNormalizer,
    RawNormalizer as RawNormalizer,
)
from .resample import Resampler as Resampler
from .rereference import ReReferencer as ReReferencer

# Expose Notch filter also in the processing module, as it is used as a data preprocessing step
from py_neuromodulation.utils.filter import NotchFilter as NotchFilter
