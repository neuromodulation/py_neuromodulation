import py_neuromodulation as pn
from py_neuromodulation import (
    nm_settings,
)

import numpy as np


def test_setting_computation_time():
    """Intantiate test for feature computation with fixed time duration.
    The number of output features should match the ratio of the sampling rate of the data and raw signal sampling rate.
    """

    data_duration_s = 5
    sampling_rate_features_hz = 200
    fs = 1000
    data = np.random.random((1, int(data_duration_s * fs)))

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)
    settings["segment_length_features_ms"] = 1000  # start afte 1 second
    settings["features"]["fft"] = False
    settings["features"]["raw_hjorth"] = True
    stream = pn.Stream(
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run()

    # test if features up till the last sample was computed
    assert (
        data_duration_s * 1000 - features["time"].iloc[-1]
    ) < 1000 / sampling_rate_features_hz

    # test that the time difference between two samples is the feature sampling rate
    assert (
        features["time"].iloc[1] - features["time"].iloc[0]
    ) == 1000 / sampling_rate_features_hz

    assert features["time"].iloc[0] == settings["segment_length_features_ms"]


def test_float_fs():
    """Change sampling rate here to be float, s.t. rounding issues are not affecting overall number of
    computed features.
    """

    data_duration_s = 5
    sampling_rate_features_hz = 200
    fs = 1111.111
    data = np.random.random((1, int(data_duration_s * fs)))

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)
    settings["segment_length_features_ms"] = 333  # start afte 1 second

    settings["features"]["fft"] = False
    settings["features"]["raw_hjorth"] = True
    stream = pn.Stream(
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run()

    # test if features up till the last sample was computed
    assert (
        data_duration_s * 1000 - features["time"].iloc[-1]
    ) < 1000 / sampling_rate_features_hz

    assert (
        features["time"].iloc[1] - features["time"].iloc[0]
    ) == 1000 / sampling_rate_features_hz

    assert features["time"].iloc[0] == settings["segment_length_features_ms"]
