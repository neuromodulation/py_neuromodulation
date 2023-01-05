"""Module for handling settings."""
import pathlib

import pandas as pd

from py_neuromodulation import nm_normalization, nm_IO


def get_default_settings() -> dict:
    """Read default settings from file."""
    settings_path = str(
        pathlib.Path(__file__).parent.resolve() / "nm_settings.json"
    )
    return nm_IO.read_settings(settings_path)


def reset_settings(
    settings: dict,
) -> dict:
    for f in settings["features"]:
        settings["features"][f] = False
    settings["preprocessing"] = []
    for f in settings["postprocessing"]:
        settings["postprocessing"][f] = False
    return settings


def set_settings_fast_compute(
    settings: dict,
) -> dict:
    settings = reset_settings(settings)
    settings["features"]["fft"] = True
    settings["preprocessing"] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]
    settings["postprocessing"]["feature_normalization"] = True
    settings["postprocessing"]["project_cortex"] = False
    settings["postprocessing"]["project_subcortex"] = False
    return settings


def test_settings(
    settings: dict,
    nm_channels: pd.DataFrame,
    verbose: bool = True,
) -> None:
    """Test if settings are specified correctly in nm_settings.json
    Parameters
    ----------
    settings: dict
        settings to tests
    verbose: boolean
        set to True if feedback is desired.
    Returns
    -------
    None
    """
    s = settings

    preprocessing_order = s["preprocessing"]

    assert isinstance(s["sampling_rate_features_hz"], (float, int))
    if s["postprocessing"]["project_cortex"] is True:
        assert isinstance(
            s["project_cortex_settings"]["max_dist_mm"], (float, int)
        )
    if s["postprocessing"]["project_subcortex"] is True:
        assert isinstance(
            s["project_subcortex_settings"]["max_dist_mm"], (float, int)
        )

    assert (
        isinstance(value, bool) for value in s["features"].values()
    ), "features must be a boolean value."
    assert any(
        value is True for value in s["features"].values()
    ), "Set at least one features to True."
    if "raw_resampling" in preprocessing_order:
        assert isinstance(
            s["raw_resampling_settings"]["resample_freq_hz"], (float, int)
        )
    if s["preprocessing"]["raw_normalization"] is True:
        assert isinstance(
            s["raw_normalization_settings"]["normalization_time_s"],
            (float, int),
        )

        assert (
            isinstance(value, bool)
            for value in s["raw_normalization_settings"][
                "normalization_method"
            ].values()
        )
        # Check if only one value is set to true.
        assert (
            sum(
                s["raw_normalization_settings"]["normalization_method"].values()
            )
            == 1
        ), "Please set only one method in raw normalization settings to true"

        assert isinstance(
            s["raw_normalization_settings"]["clip"], (float, int, bool)
        )

    if s["postprocessing"]["feature_normalization"] is True:
        assert isinstance(
            s["feature_normalization_settings"]["normalization_time_s"],
            (float, int),
        )

        assert isinstance(
            s["feature_normalization_settings"]["clip"], (float, int, bool)
        )
        assert (
            isinstance(value, bool)
            for value in s["feature_normalization_settings"][
                "normalization_method"
            ].values()
        )
        # Check if only one value is set to true.
        assert (
            sum(
                s["feature_normalization_settings"][
                    "normalization_method"
                ].values()
            )
            == 1
        ), "Please set only one method in feature normalization settings to true"

    if (
        s["bandpass_filter_settings"]["kalman_filter"] is True
        or s["stft_settings"]["kalman_filter"]
        or s["fft_settings"]["kalman_filter"]
    ):
        assert isinstance(s["kalman_filter_settings"]["Tp"], (float, int))
        assert isinstance(s["kalman_filter_settings"]["sigma_w"], (float, int))
        assert isinstance(s["kalman_filter_settings"]["sigma_v"], (float, int))
        assert s["kalman_filter_settings"][
            "frequency_bands"
        ], "No frequency bands specified for Kalman filter."
        assert isinstance(
            s["kalman_filter_settings"]["frequency_bands"], list
        ), "Frequency bands for Kalman filter must be specified as a list."
        assert (
            item
            in s["bandpass_filter_settings"]["frequency_ranges_hz"].values()
            for item in s["kalman_filter_settings"]["frequency_bands"]
        ), (
            "Frequency bands for Kalman filter must also be specified in "
            "bandpass_filter_settings."
        )
    if s["features"]["bandpass_filter"] is True:
        assert isinstance(s["frequency_ranges_hz"], dict)
        assert (
            isinstance(value, list)
            for value in s["frequency_ranges_hz"].values()
        )
        assert (len(value) == 2 for value in s["frequency_ranges_hz"].values())
        assert (
            isinstance(value[0], list)
            for value in s["frequency_ranges_hz"].values()
        )
        assert (
            len(value[0]) == 2 for value in s["frequency_ranges_hz"].values()
        )
        assert (
            isinstance(value[1], (float, int))
            for value in s["frequency_ranges_hz"].values()
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
    if s["features"]["sharpwave_analysis"] is True:
        for filter_range in s["sharpwave_analysis_settings"][
            "filter_ranges_hz"
        ]:
            assert isinstance(
                filter_range[0],
                (int, float),
            )
            assert isinstance(
                filter_range[1],
                (int, float),
            )
            assert filter_range[1] > filter_range[0]
        # check if all features are also enbled via an estimator
        used_features = list()
        for feature_name, val in s["sharpwave_analysis_settings"][
            "sharpwave_features"
        ].items():
            if val is True:
                used_features.append(feature_name)
                fun_names = []
        for used_feature in used_features:
            estimator_list_feature = (
                []
            )  # one feature can have multiple estimators
            for estimator, est_features in s["sharpwave_analysis_settings"][
                "estimator"
            ].items():
                if est_features is not None:
                    for est_feature in est_features:
                        if used_feature == est_feature:
                            estimator_list_feature.append(estimator)
            assert (
                len(estimator_list_feature) > 0
            ), f"add estimator key for {used_feature}"
            fun_names.append(estimator_list_feature)

    if s["features"]["coherence"] is True:
        assert (
            ch_coh in nm_channels.name for ch_coh in s["coherence"]["channels"]
        )

        assert (
            f_band_coh in s["frequency_ranges_hz"]
            for f_band_coh in s["coherence"]["frequency_bands"]
        )

    if verbose:
        print("No Error occurred when testing the settings.")
    return
