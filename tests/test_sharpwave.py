from pydantic import ValidationError
import pytest
import numpy as np

from py_neuromodulation import nm_sharpwaves, NMSettings


def init_sw_settings() -> NMSettings:
    settings = NMSettings.get_default()
    settings.features.sharpwave_analysis = True
    return settings


def test_sharpwaveinit_wrong_peak_param():
    settings = init_sw_settings()
    with pytest.raises(ValidationError):
        settings.sharpwave_analysis_settings.sharpwave_features.peak_left = 5

        nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names=[], sfreq=1000)


def test_sharpwaveinit_wrong_filter_range():
    """ Test sharpwave initialization with frequency higher than sampling freq"""
    settings = init_sw_settings()
    with pytest.raises(Exception):
        settings.sharpwave_analysis_settings.filter_ranges_hz = [[5, 1200]]

        nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names=[], sfreq=1000)


def test_sharpwaveinit_missing_estimator():
    """ Test sharpwave initialization with empty feature list (must fail to validate)"""

    settings = init_sw_settings()
    with pytest.raises(ValidationError):
        settings.sharpwave_analysis_settings.sharpwave_features.prominence = True
        settings.sharpwave_analysis_settings.estimator["mean"] = []
        settings.sharpwave_analysis_settings.estimator["median"] = []
        settings.sharpwave_analysis_settings.estimator["max"] = []
        settings.sharpwave_analysis_settings.estimator["min"] = []
        settings.sharpwave_analysis_settings.estimator["var"] = []

        nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names=[], sfreq=1000)


def test_sharpwaveinit_correct_featurelist():
    """ Test sharpwave initialization with correct feature list"""
    settings = init_sw_settings()
    settings.sharpwave_analysis_settings.sharpwave_features.prominence = True
    settings.sharpwave_analysis_settings.sharpwave_features.interval = True
    settings.sharpwave_analysis_settings.sharpwave_features.sharpness = True
    settings.sharpwave_analysis_settings.estimator["mean"] = [
        "prominence",
        "interval",
    ]
    settings.sharpwave_analysis_settings.estimator["median"] = ["sharpness"]
    settings.sharpwave_analysis_settings.estimator["max"] = []
    settings.sharpwave_analysis_settings.estimator["min"] = []
    settings.sharpwave_analysis_settings.estimator["var"] = []

    nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names=[], sfreq=1000)


def test_prominence_features():
    settings = init_sw_settings()
    sfreq = 1000
    ch_names = ["ch1", "ch2", "ch3", "ch4"]

    # Reset feataures
    settings.sharpwave_analysis_settings.disable_all_features()

    settings.sharpwave_analysis_settings.sharpwave_features.prominence = True
    settings.sharpwave_analysis_settings.estimator["max"] = ["prominence"]

    settings.sharpwave_analysis_settings.filter_ranges_hz = [(5, 80)]

    sw = nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names, sfreq)

    data = np.zeros([len(ch_names), sfreq])
    data[0, 500] = 1
    data[1, 500] = 2
    data[2, 500] = 3
    data[3, 500] = 4

    features = sw.calc_feature(data, {})

    assert (
        features["ch4_Sharpwave_Max_prominence_range_5_80"]
        > features["ch3_Sharpwave_Max_prominence_range_5_80"]
        > features["ch2_Sharpwave_Max_prominence_range_5_80"]
        > features["ch1_Sharpwave_Max_prominence_range_5_80"]
    ), "prominence features were not calculated correctly"


def test_interval_feature():
    settings = init_sw_settings()
    sfreq = 1000
    ch_names = ["ch1", "ch2", "ch3", "ch4"]

    # Reset feataures
    settings.sharpwave_analysis_settings.disable_all_features()

    settings.sharpwave_analysis_settings.sharpwave_features.interval = True
    settings.sharpwave_analysis_settings.estimator["max"] = ["interval"]

    # the filter cannot be too high, since adjacent ripples will be detected as peaks
    settings.sharpwave_analysis_settings.filter_ranges_hz = [[5, 200]]

    sw = nm_sharpwaves.SharpwaveAnalyzer(settings, ch_names, sfreq)

    data = np.zeros([len(ch_names), sfreq])
    for i in np.arange(0, 1000, 100):
        data[0, i] = 1
    for i in np.arange(0, 1000, 200):
        data[1, i] = 1
    for i in np.arange(0, 1000, 300):
        data[2, i] = 1
    for i in np.arange(0, 1000, 400):
        data[3, i] = 1

    features = sw.calc_feature(data, {})

    print(features.keys())
    assert (
        features["ch1_Sharpwave_Max_interval_range_5_200"]
        < features["ch2_Sharpwave_Max_interval_range_5_200"]
        < features["ch3_Sharpwave_Max_interval_range_5_200"]
        < features["ch4_Sharpwave_Max_interval_range_5_200"]
    ), "interval features were not calculated correctly"
