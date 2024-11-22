from os import PathLike
from math import isnan
from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, model_validator
from .pydantic_extensions import NMBaseModel, NMSequenceModel

from collections.abc import Sequence
from datetime import datetime


if TYPE_CHECKING:
    import numpy as np
    from py_neuromodulation import NMSettings

###################################
########## TYPE ALIASES  ##########
###################################

_PathLike = str | PathLike

FeatureName = Literal[
    "raw_hjorth",
    "return_raw",
    "bandpass_filter",
    "stft",
    "fft",
    "welch",
    "sharpwave_analysis",
    "fooof",
    "nolds",
    "coherence",
    "bursts",
    "linelength",
    "mne_connectivity",
    "bispectrum",
]

PreprocessorName = Literal[
    "preprocessing_filter",
    "notch_filter",
    "raw_resampling",
    "re_referencing",
    "raw_normalization",
]

NormMethod = Literal[
    "mean",
    "median",
    "zscore",
    "zscore-median",
    "quantile",
    "power",
    "robust",
    "minmax",
]


###################################
######## PROTOCOL CLASSES  ########
###################################


class NMFeature:
    def __init__(
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: int | float
    ) -> None: ...

    def calc_feature(self, data: "np.ndarray") -> dict:
        """
        Feature calculation method. Each method needs to loop through all channels

        Parameters
        ----------
        data : 'np.ndarray'
            (channels, time)

        Returns
        -------
        dict
        """
        ...


class NMPreprocessor:
    def process(self, data: "np.ndarray") -> "np.ndarray": ...


class PreprocessorList(NMSequenceModel[list[PreprocessorName]]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Useless contructor to prevent linter from complaining
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class FrequencyRange(NMSequenceModel[tuple[float, float]]):
    """Frequency range as (low, high) tuple"""

    __aliases__ = {
        0: ["frequency_low_hz", "low_frequency_hz"],
        1: ["frequency_high_hz", "high_frequency_hz"],
    }

    # Useless contructor to prevent linter from complaining
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @model_validator(mode="after")
    def validate_range(self):
        low, high = self.root
        if not (isnan(low) or isnan(high)):
            assert high > low, "High frequency must be greater than low frequency"
        return self

    # Alias properties
    @property
    def frequency_low_hz(self) -> float:
        """Lower frequency bound in Hz"""
        return self.root[0]

    @property
    def frequency_high_hz(self) -> float:
        """Upper frequency bound in Hz"""
        return self.root[1]


class BoolSelector(NMBaseModel):
    def get_enabled(self):
        return [
            f
            for f in self.model_fields.keys()
            if (isinstance(self[f], bool) and self[f])
        ]

    def enable_all(self):
        for f in self.model_fields.keys():
            if isinstance(self[f], bool):
                self[f] = True

    def disable_all(self):
        for f in self.model_fields.keys():
            if isinstance(self[f], bool):
                self[f] = False

    def __iter__(self):  # type: ignore
        return iter(self.model_dump().keys())

    @classmethod
    def list_all(cls):
        return list(cls.model_fields.keys())

    @classmethod
    def print_all(cls):
        for f in cls.list_all():
            print(f)


#################
### GUI TYPES ###
#################


class FileInfo(BaseModel):
    name: str
    path: str
    dir: str
    is_directory: bool
    size: int
    created_at: datetime
    modified_at: datetime
