from os import PathLike
from math import isnan
from typing import Literal, TYPE_CHECKING, Any
from pydantic import BaseModel, ConfigDict, model_validator
from .pydantic_extensions import NMBaseModel, NMSequenceModel, NMField
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


class PreprocessorList(NMSequenceModel[list[PREPROCESSOR_NAME]]):
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


################################################
### Generic Pydantic models for the frontend ###
################################################


class UniqueStringSequence(NMSequenceModel[list[str]]):
    """
    A sequence of strings where:
    - Values must come from a predefined set
    - Each value can only appear once
    - Order is preserved
    """

    @property
    @abstractmethod
    def valid_values(self) -> list[str]:
        """Each subclass must implement this to provide its valid values"""
        raise NotImplementedError

    def __init__(self, **data):
        valid_values = data.pop("valid_values", [])
        super().__init__(**data)
        object.__setattr__(self, "valid_values", valid_values)

    @model_validator(mode="after")
    def validate_sequence(self):
        seen = set()
        validated = []
        for item in self.root:
            if item not in seen and item in self.valid_values:
                seen.add(item)
                validated.append(item)
        self.root = validated
        return self

    def serialize_with_metadata(self) -> dict[str, Any]:
        result = super().serialize_with_metadata()
        result["__valid_values__"] = self.valid_values
        return result


class DependentKeysList(NMSequenceModel[list[str]]):
    """
    A list of strings where valid values are keys from another settings field
    """

    root: list[str] = NMField(default_factory=list)
    source_dict: dict[str, Any] = NMField(default_factory=dict, exclude=True)

    def __init__(self, **data):
        source_dict = data.pop("source_dict", {})
        super().__init__(**data)
        object.__setattr__(self, "source_dict", source_dict)

    @model_validator(mode="after")
    def validate_keys(self):
        valid_keys = set(self.source_dict.keys())
        seen = set()
        validated = []
        for item in self.root:
            if item not in seen and item in valid_keys:
                seen.add(item)
                validated.append(item)
        self.root = validated
        return self

    def serialize_with_metadata(self) -> dict[str, Any]:
        result = super().serialize_with_metadata()
        result["__valid_values__"] = list(self.source_dict.keys())
        result["__dependent__"] = True  # Indicates this needs dynamic updating
        return result


class StringPairsList(NMSequenceModel[list[tuple[str, str]]]):
    """
    A list of string pairs where values must come from predetermined lists
    """

    root: list[tuple[str, str]] = NMField(default_factory=list)
    valid_first: list[str] = NMField(default_factory=list, exclude=True)
    valid_second: list[str] = NMField(default_factory=list, exclude=True)

    def __init__(self, **data):
        valid_first = data.pop("valid_first", [])
        valid_second = data.pop("valid_second", [])
        super().__init__(**data)
        object.__setattr__(self, "valid_first", valid_first)
        object.__setattr__(self, "valid_second", valid_second)

    @model_validator(mode="after")
    def validate_pairs(self):
        validated = [
            (first, second)
            for first, second in self.root
            if first in self.valid_first and second in self.valid_second
        ]
        self.root = validated
        return self

    def serialize_with_metadata(self) -> dict[str, Any]:
        result = super().serialize_with_metadata()
        result["__valid_first__"] = self.valid_first
        result["__valid_second__"] = self.valid_second
        return result


# class LiteralValue(NMValueModel[str]):
#     """
#     A string field that must be one of a predefined set of literals
#     """

#     valid_values: list[str] = NMField(default_factory=list, exclude=True)

#     def __init__(self, **data):
#         valid_values = data.pop("valid_values", [])
#         super().__init__(**data)
#         object.__setattr__(self, "valid_values", valid_values)

#     @model_validator(mode="after")
#     def validate_value(self):
#         if self.root not in self.valid_values:
#             raise ValueError(f"Value must be one of: {self.valid_values}")
#         return self

#     def serialize_with_metadata(self) -> dict[str, Any]:
#         result = super().serialize_with_metadata()
#         result["__valid_values__"] = self.valid_values
#         return result


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
