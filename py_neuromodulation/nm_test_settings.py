import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from py_neuromodulation import nm_define_nmchannels

def test_settings(self, verbose=True) -> None:
    """Test if settings are specified correctly in nm_settings.json

    Parameters
    ----------
    verbose: boolean
        set to True if feedback is desired.
    Returns
    -------
    None
    """
    if isinstance(self.settings, dict):
        s = self.settings
    else:
        with open(self.settings_path, encoding="utf-8") as json_file:
            s = json.load(json_file)
        assert isinstance(s, dict)

    # assert (os.path.isdir(s["BIDS_path"]))
    # assert (os.path.isdir(s["out_path"]))
    assert isinstance(s["sampling_rate_features"], (float, int))
    if s["methods"]["project_cortex"] is True:
        assert isinstance(
            s["project_cortex_settings"]["max_dist"], (float, int)
        )
    if s["methods"]["project_subcortex"] is True:
        assert isinstance(
            s["project_subcortex_settings"]["max_dist"], (float, int)
        )
    assert (
        isinstance(value, bool) for value in s["methods"].values()
    ), "Methods must be a boolean value."
    assert any(
        value is True for value in s["methods"].values()
    ), "Set at least one method to True."
    if s["methods"]["raw_resampling"] is True:
        assert isinstance(
            s["raw_resampling_settings"]["resample_freq"], (float, int)
        )
    if s["methods"]["raw_normalization"] is True:
        assert isinstance(
            s["raw_normalization_settings"]["normalization_time"],
            (float, int),
        )
        assert s["raw_normalization_settings"]["normalization_method"] in [
            "mean",
            "median",
            "zscore",
        ]
        assert isinstance(
            s["raw_normalization_settings"]["clip"], (float, int, bool)
        )
    if s["methods"]["feature_normalization"] is True:
        assert isinstance(
            s["feature_normalization_settings"]["normalization_time"],
            (float, int),
        )
        assert s["feature_normalization_settings"][
            "normalization_method"
        ] in ["mean", "median", "zscore", "zscore-median"]
        assert isinstance(
            s["feature_normalization_settings"]["clip"], (float, int, bool)
        )
    if s["methods"]["kalman_filter"] is True:
        assert isinstance(s["kalman_filter_settings"]["Tp"], (float, int))
        assert isinstance(
            s["kalman_filter_settings"]["sigma_w"], (float, int)
        )
        assert isinstance(
            s["kalman_filter_settings"]["sigma_v"], (float, int)
        )
        assert s["kalman_filter_settings"][
            "frequency_bands"
        ], "No frequency bands specified for Kalman filter."
        assert isinstance(
            s["kalman_filter_settings"]["frequency_bands"], list
        ), "Frequency bands for Kalman filter must be specified as a list."
        assert (
            item
            in s["bandpass_filter_settings"]["frequency_ranges"].values()
            for item in s["kalman_filter_settings"]["frequency_bands"]
        ), (
            "Frequency bands for Kalman filter must also be specified in "
            "bandpass_filter_settings."
        )
    if s["methods"]["bandpass_filter"] is True:
        assert isinstance(s["frequency_ranges"], dict)
        assert (
            isinstance(value, list)
            for value in s["frequency_ranges"].values()
        )
        assert (
            len(value) == 2 for value in s["frequency_ranges"].values()
        )
        assert (
            isinstance(value[0], list)
            for value in s["frequency_ranges"].values()
        )
        assert (
            len(value[0]) == 2 for value in s["frequency_ranges"].values()
        )
        assert (
            isinstance(value[1], (float, int))
            for value in s["frequency_ranges"].values()
        )
        assert (
            isinstance(value, bool)
            for value in s["bandpass_filter_settings"][
                "bandpower_features"
            ].values()
        )
        assert any(
            value is True
            for value in s["bandpass_filter_settings"][
                "bandpower_features"
            ].values()
        ), "Set at least one bandpower_feature to True."
    if s["methods"]["sharpwave_analysis"] is True:
        assert isinstance(
            s["sharpwave_analysis_settings"]["filter_low_cutoff"],
            (int, float),
        )
        assert isinstance(
            s["sharpwave_analysis_settings"]["filter_high_cutoff"],
            (int, float),
        )
        assert (
            s["sharpwave_analysis_settings"]["filter_high_cutoff"]
            > s["sharpwave_analysis_settings"]["filter_low_cutoff"]
        )
    if verbose:
        print("No Error occurred when testing the settings.")
    return
