from typing import Protocol, TYPE_CHECKING
from inspect import getfullargspec
from typing import Type
from py_neuromodulation.nm_types import ImportDetails, get_class

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

class NMPreprocessor(Protocol):
    def __init__(self, settings: dict, sfreq: float) -> None: ...

    def process(self, data: 'np.ndarray') -> 'np.ndarray': ...


PREPROCESSOR_DICT: dict[str, ImportDetails] = {
    "preprocessing_filter": ImportDetails(
        "py_neuromodulation.nm_filter_preprocessing", "PreprocessingFilter"
    ),
    "notch_filter": ImportDetails("py_neuromodulation.nm_filter", "NotchFilter"),
    "raw_resampling": ImportDetails("py_neuromodulation.nm_resample", "Resampler"),
    "re_referencing": ImportDetails(
        "py_neuromodulation.nm_rereference", "ReReferencer"
    ),
    "raw_normalization": ImportDetails(
        "py_neuromodulation.nm_normalization", "RawNormalizer"
    ),
}

class NMPreprocessors:
    "Class for initializing and holding data preprocessing classes"

    def __init__(
        self,
        settings: dict,
        nm_channels: "pd.DataFrame",
        sfreq: float,
        line_noise: float | None = None,
    ) -> None:
        possible_arguments = {
            "sfreq": sfreq,
            "settings": settings,
            "nm_channels": nm_channels,
            "line_noise": line_noise,
        }

        for preprocessing_method in settings["preprocessing"]:
            if preprocessing_method not in PREPROCESSOR_DICT.keys():
                raise ValueError(
                    f"Invalid preprocessing method '{preprocessing_method}'. Must be one of {PREPROCESSOR_DICT.keys()}"
                )

        # Get needed preprocessor classes from settings
        preprocessor_classes: dict[str, Type[NMPreprocessor]] = {
            preprocessor_name: get_class(import_details)
            for preprocessor_name, import_details in PREPROCESSOR_DICT.items()
            if preprocessor_name in settings["preprocessing"]
        }

        # Function to instantiate preprocessor with settings
        def instantiate_preprocessor(
            preprocessor_class: Type[NMPreprocessor], preprocessor_name: str
        ) -> NMPreprocessor:
            settings_str = f"{preprocessor_name}_settings"
            # Filter out arguments that are not in the preprocessor's __init__ method
            args = {
                arg: possible_arguments[arg]
                for arg in getfullargspec(preprocessor_class).args
                if arg in possible_arguments
            }
            # Retrieve more possible arguments from settings
            args |= settings.get(settings_str, {})
            # Pass arguments to preprocessor class and return instance
            return preprocessor_class(**args)

        self.preprocessors: list[NMPreprocessor] = [
            instantiate_preprocessor(preprocessor_class, preprocessor_name)
            for preprocessor_name, preprocessor_class in preprocessor_classes.items()
        ]

    def process_data(self, data: 'np.ndarray') -> 'np.ndarray':
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)
        return data