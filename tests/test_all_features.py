import numpy as np
import pandas as pd
import pytest

from py_neuromodulation import (
    nm_settings,
    nm_stream_offline,
    nm_define_nmchannels,
    nm_stream_abc
)

def get_example_stream(test_arr: np.array) -> nm_stream_abc.PNStream:
    settings = nm_settings.get_default_settings()
    settings["features"]["raw_hjorth"] = True
    settings["features"]["return_raw"] = True
    settings["features"]["bandpass_filter"] = True
    settings["features"]["stft"] = True
    settings["features"]["fft"] = True
    settings["features"]["sharpwave_analysis"] = True
    settings["features"]["fooof"] = True
    settings["features"]["bursts"] = True
    settings["features"]["linelength"] = True
    settings["features"]["nolds"] = False
    settings["features"]["mne_connectivity"] = False
    settings["features"]["coherence"] = False

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

    stream = nm_stream_offline.Stream(
        sfreq=1000,
        nm_channels=nm_channels,
        settings=settings,
        verbose=True
    )
    return stream

def test_all_features_random_array():

    arr = np.random.random([2, 2000])
    stream = get_example_stream(arr)

    df = stream.run(arr)

    # data CAN contain None's
    assert df.shape[1] == df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).sum()

def test_all_features_zero_array():

    arr = np.zeros([2, 2000])
    stream = get_example_stream(arr)

    df = stream.run(arr)

def test_all_features_NaN_array():

    arr = np.empty([2, 2000])
    arr[:] = np.nan

    stream = get_example_stream(arr)

    df = stream.run(arr)
