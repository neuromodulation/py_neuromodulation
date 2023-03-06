import numpy as np
import pandas as pd
import pytest

from py_neuromodulation import (
    nm_settings,
    nm_stream_offline,
    nm_define_nmchannels,
    nm_stream_abc
)

def get_example_settings(test_arr: np.array) -> nm_stream_abc.PNStream:
    settings = nm_settings.set_settings_fast_compute(
        nm_settings.get_default_settings()
    )

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

    return settings, nm_channels

def test_different_sampling_rates():

    arr_test = np.random.random([2, 2000])
    settings, nm_channels = get_example_settings(arr_test)

    settings["sampling_rate_features_hz"] = 10
    stream = nm_stream_offline.Stream(
        sfreq=1000,
        nm_channels=nm_channels,
        settings=settings,
        verbose=True
    )

    df = stream.run()

