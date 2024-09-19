import numpy as np
import pandas as pd
import pytest

import py_neuromodulation as nm


def get_example_settings(test_arr: np.ndarray) -> tuple[nm.NMSettings, pd.DataFrame]:
    settings = nm.NMSettings.get_fast_compute()

    channels = nm.utils.get_default_channels_from_data(test_arr)

    return settings, channels


def test_different_sampling_rate_100Hz():
    sampling_rate_features = 100

    arr_test = np.random.random([2, 1020])
    settings, channels = get_example_settings(arr_test)

    settings.sampling_rate_features_hz = sampling_rate_features
    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_sampling_rate_100Hz",
    )

    # check the difference between time points
    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_10Hz():
    sampling_rate_features = 10

    arr_test = np.random.random([2, 1200])
    settings, channels = get_example_settings(arr_test)

    settings.sampling_rate_features_hz = sampling_rate_features
    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_sampling_rate_10Hz",
    )

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_1Hz():
    sampling_rate_features = 1

    arr_test = np.random.random([2, 3000])
    settings, channels = get_example_settings(arr_test)

    settings.sampling_rate_features_hz = sampling_rate_features
    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_sampling_rate_1Hz",
    )

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_0DOT1Hz():
    sampling_rate_features = 0.1

    arr_test = np.random.random([2, 30000])
    settings, channels = get_example_settings(arr_test)

    settings.sampling_rate_features_hz = sampling_rate_features
    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_sampling_rate_0DOT1Hz",
    )

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_wrong_initalization_of_segment_length_features_ms_and_osc_window_length():
    arr_test = np.random.random([2, 1200])
    settings, channels = get_example_settings(arr_test)

    settings.segment_length_features_ms = 800
    settings.fft_settings.windowlength_ms = 1000

    with pytest.raises(Exception):
        nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)


def test_different_segment_lengths():
    segment_length_features_ms = 800

    arr_test = np.random.random([2, 1200])
    settings, channels = get_example_settings(arr_test)

    settings.segment_length_features_ms = segment_length_features_ms
    settings.fft_settings.windowlength_ms = segment_length_features_ms

    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df_seglength_800 = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_segment_lengths_800",
    )

    segment_length_features_ms = 1000

    arr_test = np.random.random([2, 1200])
    settings, channels = get_example_settings(arr_test)

    settings.segment_length_features_ms = segment_length_features_ms
    settings.fft_settings.windowlength_ms = segment_length_features_ms

    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)

    df_seglength_1000 = stream.run(
        arr_test,
        out_path_root="./test_data",
        folder_name="test_different_segment_lengths_1000",
    )
    # check the difference between time points

    print(df_seglength_1000.columns)
    assert (
        df_seglength_1000.iloc[0]["ch0_avgref_fft_theta_mean"]
        != df_seglength_800.iloc[0]["ch0_avgref_fft_theta_mean"]
    )
