from os import PathLike
from typing import NamedTuple, Type, Any, Literal
from importlib import import_module
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from dataclasses import fields

_PathLike = str | PathLike


class ImportDetails(NamedTuple):
    module_name: str
    class_name: str


@dataclass
class FrequencyRange:
    frequency_low_hz: float = Field(default=0, gt=0)
    frequency_high_hz: float = Field(default=0, gt=0)

    def __getitem__(self, item: int):
        match item:
            case 0:
                return self.frequency_low_hz
            case 1:
                return self.frequency_high_hz
            case _:
                raise IndexError(f"Index {item} out of range")

    @model_validator(mode="after")
    def validate_range(self):
        assert (
            self.frequency_high_hz > self.frequency_low_hz
        ), "Frequency high must be greater than frequency low"
        return self


@dataclass
class FeatureSelector:
    def get_enabled(self):
        return [f.name for f in fields(self) if getattr(self, f.name)]

    @classmethod
    def list_all(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def print_all(cls):
        for f in cls.list_all():
            print(f)


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


def get_class(module_details: ImportDetails) -> Type[Any]:
    return getattr(import_module(module_details.module_name), module_details.class_name)
    # return getattr(import_module("py_neuromodulation." + module_details.module_name), module_details.class_name)
