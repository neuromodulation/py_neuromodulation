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
        experiment_name="test_setting_computation_time",
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run(out_dir="./test_data")

    # test if features up till the last sample was computed
    assert features.time.iloc[-1] == data_duration_s * fs

    # test that the time difference between two samples is the feature sampling rate
    assert (
        features.time.iloc[1] - features.time.iloc[0]
    ) == fs / sampling_rate_features_hz

    assert features.time.iloc[0] == settings.segment_length_features_ms


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
        experiment_name="test_float_fs",
        sfreq=fs,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
        settings=settings,
    )

    features = stream.run(out_dir="./test_data")

    # test that the time difference between two samples is the feature sampling rate
    assert (
        features.time.iloc[1] - features.time.iloc[0]
    ) == 1000 / sampling_rate_features_hz

    # the timing of the first sample cannot be directly inferred from the segment length
    # the samples are computed based on rounding to full indices of the original data
