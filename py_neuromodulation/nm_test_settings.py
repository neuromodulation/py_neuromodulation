import pandas as pd
from py_neuromodulation import nm_normalization


def test_settings(
    settings: dict,
    nm_channel: pd.DataFrame,
    verbose=True,
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

    assert isinstance(s["sampling_rate_features_hz"], (float, int))
    if s["postprocessing"]["project_cortex"] is True:
        assert isinstance(
            s["project_cortex_settings"]["max_dist_cm"], (float, int)
        )
    if s["postprocessing"]["project_subcortex"] is True:
        assert isinstance(
            s["project_subcortex_settings"]["max_dist_cm"], (float, int)
        )
    enabled_methods = [
        m
        for m in s["preprocessing"]
        if "order" not in m and s["preprocessing"][m] is True
    ]
    for preprocess_method in s["preprocessing"]["preprocessing_order"]:
        assert (
            preprocess_method in enabled_methods
        ), "Enabled Preprocessing methods need to be listed in preprocesssing_order"

    assert (
        isinstance(value, bool) for value in s["features"].values()
    ), "features must be a boolean value."
    assert any(
        value is True for value in s["features"].values()
    ), "Set at least one features to True."
    if s["preprocessing"]["raw_resampling"] is True:
        assert isinstance(
            s["raw_resampling_settings"]["resample_freq_hz"], (float, int)
        )
    if s["preprocessing"]["raw_normalization"] is True:
        assert isinstance(
            s["raw_normalization_settings"]["normalization_time_s"],
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
    if s["postprocessing"]["feature_normalization"] is True:
        assert isinstance(
            s["feature_normalization_settings"]["normalization_time_s"],
            (float, int),
        )
        assert s["feature_normalization_settings"]["normalization_method"] in [
            e.value for e in nm_normalization.NORM_METHODS
        ]
        assert isinstance(
            s["feature_normalization_settings"]["clip"], (float, int, bool)
        )
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
            ch_coh in nm_channel.name for ch_coh in s["coherence"]["channels"]
        )

        assert (
            f_band_coh in s["frequency_ranges_hz"]
            for f_band_coh in s["coherence"]["frequency_bands"]
        )

    if verbose:
        print("No Error occurred when testing the settings.")
    return
