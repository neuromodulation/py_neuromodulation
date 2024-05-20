from os import PathLike
from math import isnan
from typing import NamedTuple, Type, Any, Literal
from importlib import import_module
from pydantic import Field, model_validator, BaseModel

_PathLike = str | PathLike


class NMBaseModel(BaseModel):
    def __init__(self, *args, **kwargs) -> None:
        if kwargs:
            super().__init__(**kwargs)
        else:
            field_names = list(self.model_fields.keys())
            kwargs = {}
            for i in range(len(args)):
                kwargs[field_names[i]] = args[i]
            super().__init__(**kwargs)
            
            
    def __str__(self):
        return print(str(self.dict()))


class ImportDetails(NamedTuple):
    module_name: str
    class_name: str


class FrequencyRange(NMBaseModel):
    frequency_low_hz: float = Field(default=0, gt=0)
    frequency_high_hz: float = Field(default=0, gt=0)

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

    @model_validator(mode="after")
    def validate_range(self):
        if not (isnan(self.frequency_high_hz) or isnan(self.frequency_low_hz)):
            assert (
                self.frequency_high_hz > self.frequency_low_hz
            ), "Frequency high must be greater than frequency low"
        return self

    @model_validator(mode="before")
    @classmethod
    def check_freq_ranges(cls, input):
        if not (isinstance(input, cls) or isinstance(input, dict)):
            if len(input) == 2:
                return {"frequency_low_hz": input[0], "frequency_high_hz": input[1]}
            else:
                raise ValueError(
                    "Value for FrequencyRange must be a diciontary,"
                    "a FrequencyRange object or an iterable of 2 numeric values,"
                    f"but got {input} instead."
                )
        else:
            return input

class FeatureSelector(BaseModel):
    def get_enabled(self):
        return [f for f in self.model_fields.keys() if getattr(self, f)]

    @classmethod
    def list_all(cls):
        return list(cls.model_fields.keys())

    @classmethod
    def print_all(cls):
        for f in cls.list_all():
            print(f)
    
    @classmethod 
    def get_fields(cls):
        return cls.model_fields


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
    # return getattr(import_module(module_details.module_name), module_details.class_name)
    return getattr(import_module("py_neuromodulation." + module_details.module_name), module_details.class_name)
