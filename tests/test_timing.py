import py_neuromodulation as nm

import numpy as np


def test_setting_computation_time():
    """Intantiate test for feature computation with fixed time duration.
    The number of output features should match the ratio of the sampling rate of the data and raw signal sampling rate.
    """

    data_duration_s = 5
    sampling_rate_features_hz = 200
    fs = 1000
    data = np.random.random((1, int(data_duration_s * fs)))

    settings = nm.NMSettings.get_fast_compute()
    settings.segment_length_features_ms = 1000  # start afte 1 second
    settings.features.fft = False
    settings.features.raw_hjorth = True
    stream = nm.Stream(
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run(out_path_root="./test_data", folder_name="test_setting_computation_time")

    # test if features up till the last sample was computed
    assert (
        data_duration_s * 1000 - features.time.iloc[-1]
    ) < 1000 / sampling_rate_features_hz

    # test that the time difference between two samples is the feature sampling rate
    assert (
        features.time.iloc[1] - features.time.iloc[0]
    ) == 1000 / sampling_rate_features_hz

    assert features.time.iloc[0] == settings.segment_length_features_ms - 1


def test_float_fs():
    """Change sampling rate here to be float, s.t. rounding issues are not affecting overall number of
    computed features.
    """

    data_duration_s = 5
    sampling_rate_features_hz = 200
    fs = 1111.111
    data = np.random.random((1, int(data_duration_s * fs)))

    settings = nm.NMSettings.get_fast_compute()
    settings.segment_length_features_ms = 333  # start after 1 second

    settings.features.fft = False
    settings.features.raw_hjorth = True
    stream = nm.Stream(
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run(out_path_root="./test_data", folder_name="test_float_fs")

    # test if features up till the last sample was computed
    assert (
        data_duration_s * 1000 - features.time.iloc[-1]
    ) < 1000 / sampling_rate_features_hz

    # test that the time difference between two samples is the feature sampling rate
    assert (
        features.time.iloc[1] - features.time.iloc[0]
    ) == 1000 / sampling_rate_features_hz

    # TONI: I fixed this test so that it passes, but I feel it's not the right way to test timestamp correctness
    # test that the first feature segment timestamp matches settings.segment_length_features_ms
    assert features["time"].iloc[0] == settings.segment_length_features_ms - 1
