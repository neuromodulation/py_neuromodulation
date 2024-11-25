from typing import TYPE_CHECKING, Type
from py_neuromodulation.utils.types import PREPROCESSOR_NAME, NMPreprocessor

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from py_neuromodulation.stream.settings import NMSettings

PREPROCESSOR_DICT: dict[PREPROCESSOR_NAME, str] = {
    "preprocessing_filter": "PreprocessingFilter",
    "notch_filter": "NotchFilter",
    "raw_resampling": "Resampler",
    "re_referencing": "ReReferencer",
    "raw_normalization": "RawNormalizer",
}


class DataPreprocessor:
    "Class for initializing and holding data preprocessing classes"

    def __init__(
        self,
        settings: "NMSettings",
        channels: "pd.DataFrame",
        sfreq: float,
        line_noise: float | None = None,
    ) -> None:
        from importlib import import_module
        from inspect import getfullargspec

        possible_arguments = {
            "sfreq": sfreq,
            "settings": settings,
            "channels": channels,
            "line_noise": line_noise,
        }

        for preprocessing_method in settings.preprocessing:
            if preprocessing_method not in PREPROCESSOR_DICT.keys():
                raise ValueError(
                    f"Invalid preprocessing method '{preprocessing_method}'. Must be one of {PREPROCESSOR_DICT.keys()}"
                )

        # Get needed preprocessor classes from settings
        preprocessor_classes: dict[str, Type[NMPreprocessor]] = {
            preprocessor_name: getattr(
                import_module("py_neuromodulation.processing"), class_name
            )
            for preprocessor_name, class_name in PREPROCESSOR_DICT.items()
            if preprocessor_name in settings.preprocessing
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
            args |= getattr(settings, settings_str, {})
            # Pass arguments to preprocessor class and return instance
            return preprocessor_class(**args)

        self.preprocessors: list[NMPreprocessor] = [
            instantiate_preprocessor(preprocessor_class, preprocessor_name)
            for preprocessor_name, preprocessor_class in preprocessor_classes.items()
        ]

    def process_data(self, data: "np.ndarray") -> "np.ndarray":
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)
        return data
