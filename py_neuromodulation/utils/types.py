from os import PathLike
from math import isnan
from typing import Literal, TYPE_CHECKING, Any
from pydantic import BaseModel, model_validator, Field
from .pydantic_extensions import NMBaseModel, NMField
from abc import abstractmethod

from collections.abc import Sequence
from datetime import datetime


if TYPE_CHECKING:
    import numpy as np
    from py_neuromodulation import NMSettings

###################################
########## TYPE ALIASES  ##########
###################################

_PathLike = str | PathLike

FEATURE_NAME = Literal[
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

PREPROCESSOR_NAME = Literal[
    "preprocessing_filter",
    "notch_filter",
    "raw_resampling",
    "re_referencing",
    "raw_normalization",
]

NORM_METHOD = Literal[
    "mean",
    "median",
    "zscore",
    "zscore-median",
    "quantile",
    "power",
    "robust",
    "minmax",
]


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


class FrequencyRange(NMBaseModel):
    frequency_low_hz: float = Field(gt=0)
    frequency_high_hz: float = Field(gt=0)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: int):
        match item:
            case 0:
                return self.frequency_low_hz
            case 1:
                return self.frequency_high_hz
            case _:
                raise IndexError(f"Index {item} out of range")

    def as_tuple(self) -> tuple[float, float]:
        return (self.frequency_low_hz, self.frequency_high_hz)

    def __iter__(self):  # type: ignore
        return iter(self.as_tuple())

    @model_validator(mode="after")
    def validate_range(self):
        if not (isnan(self.frequency_high_hz) or isnan(self.frequency_low_hz)):
            assert self.frequency_high_hz > self.frequency_low_hz, (
                "Frequency high must be greater than frequency low"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_input(cls, input):
        """Pydantic validator to convert the input to a dictionary when passed as a list
        as we have it by default in the default_settings.yaml file
        For example, [1,2] will be converted to {"frequency_low_hz": 1, "frequency_high_hz": 2}
        """
        match input:
            case dict() if "frequency_low_hz" in input and "frequency_high_hz" in input:
                return input
            case Sequence() if len(input) == 2:
                return {"frequency_low_hz": input[0], "frequency_high_hz": input[1]}
            case _:
                raise ValueError(
                    "Value for FrequencyRange must be a dictionary, "
                    "or a sequence of 2 numeric values, "
                    f"but got {input} instead."
                )


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