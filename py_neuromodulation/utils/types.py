from os import PathLike
from math import isnan
from typing import Any, Literal, Protocol, TYPE_CHECKING, runtime_checkable
from pydantic import ConfigDict, Field, model_validator, BaseModel
from pprint import pformat
from collections.abc import Sequence

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


@runtime_checkable
class NMFeature(Protocol):
    def __init__(
        self, settings: "NMSettings", ch_names: Sequence[str], sfreq: int | float
    ) -> None: ...

    def calc_feature(self, data: "np.ndarray", features_compute: dict) -> dict:
        """
        Feature calculation method. Each method needs to loop through all channels

        Parameters
        ----------
        data : 'np.ndarray'
            (channels, time)
        features_compute : dict

        Returns
        -------
        dict
        """
        ...


class NMPreprocessor(Protocol):
    def __init__(self, sfreq: float, settings: "NMSettings") -> None: ...

    def process(self, data: "np.ndarray") -> "np.ndarray": ...


###################################
######## PYDANTIC CLASSES  ########
###################################


class NMBaseModel(BaseModel):
    model_config = ConfigDict(validate_assignment=False, extra="allow")

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
        return pformat(self.model_dump())

    def __repr__(self):
        return pformat(self.model_dump())

    def validate(self) -> Any:  # type: ignore
        return self.model_validate(self.model_dump())

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)


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

    def __iter__(self):  # type: ignore
        return iter(self.as_tuple())

    @model_validator(mode="after")
    def validate_range(self):
        if not (isnan(self.frequency_high_hz) or isnan(self.frequency_low_hz)):
            assert (
                self.frequency_high_hz > self.frequency_low_hz
            ), "Frequency high must be greater than frequency low"
        return self

    @classmethod
    def create_from(cls, input) -> "FrequencyRange":
        match input:
            case FrequencyRange():
                return input
            case dict() if "frequency_low_hz" in input and "frequency_high_hz" in input:
                return FrequencyRange(
                    input["frequency_low_hz"], input["frequency_high_hz"]
                )
            case Sequence() if len(input) == 2:
                return FrequencyRange(input[0], input[1])
            case _:
                raise ValueError("Invalid input for FrequencyRange creation.")

    @model_validator(mode="before")
    @classmethod
    def check_input(cls, input):
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

    @classmethod
    def get_fields(cls):
        return cls.model_fields
