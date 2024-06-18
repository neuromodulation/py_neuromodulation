import numpy as np
import pandas as pd

from py_neuromodulation import (
    NMSettings,
    nm_stream,
    nm_define_nmchannels,
)


def get_example_settings(test_arr: np.ndarray) -> tuple[NMSettings, pd.DataFrame]:
    settings = NMSettings.get_fast_compute()

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

    return settings, nm_channels


def test_label_add_single_target():
    sampling_rate_features = 10
    arr_test = np.random.random([2, 5000])
    target_arr = np.arange(0, 5000, 1)
    arr_test[1, :] = target_arr

    settings, nm_channels = get_example_settings(arr_test)

    nm_channels["target"] = [0, 1]
    nm_channels["used"] = [1, 0]
    nm_channels.loc[1, "name"] = "target_ch"

    settings.sampling_rate_features_hz = sampling_rate_features

    stream = nm_stream.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    assert df[
        "target_ch"
    ].is_monotonic_increasing, "Not all target values are ascending"

    # check if all "target_ch" values are in the target_arr
    assert all(
        [i in target_arr for i in df["target_ch"]]
    ), "Not all target values are were estimated correctly from the simulated target array"


def test_label_add_multidimensional_target():
    sampling_rate_features = 10
    arr_test = np.random.random([4, 5000])
    target_arr = np.arange(0, 5000, 1)
    target_arr = np.tile(target_arr, (2, 1))
    arr_test[[0, 2], :] = target_arr

    settings, nm_channels = get_example_settings(arr_test)

    nm_channels["target"] = [1, 0, 1, 0]
    nm_channels["used"] = [0, 1, 0, 1]
    nm_channels.loc[0, "name"] = "target_ch_0"
    nm_channels.loc[2, "name"] = "target_ch_1"

    settings.sampling_rate_features_hz = sampling_rate_features

    stream = nm_stream.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    for target_ch in ["target_ch_0", "target_ch_1"]:
        assert df[
            target_ch
        ].is_monotonic_increasing, "Not all target values are ascending"

        # check if all "target_ch" values are in the target_arr
        assert all(
            [i in target_arr for i in df[target_ch]]
        ), "Not all target values are were estimated correctly from the simulated target array"


def test_label_add_no_target():
    sampling_rate_features = 10
    arr_test = np.random.random([4, 5000])

    settings, nm_channels = get_example_settings(arr_test)

    settings.sampling_rate_features_hz = sampling_rate_features

    stream = nm_stream.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    assert all([col.startswith("ch") or col.startswith("time") for col in df.columns])
