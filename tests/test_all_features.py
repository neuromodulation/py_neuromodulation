import numpy as np

from py_neuromodulation import (
    nm_settings,
    nm_define_nmchannels,
    Stream,
)


def get_example_stream(test_arr: np.ndarray) -> Stream:
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

    stream = Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )
    return stream


def test_all_features_random_array():
    """This test runs's through all enabled features, and check's if they break"""
    np.random.seed(0)
    arr = np.random.random([2, 2000])
    stream = get_example_stream(arr)

    df = stream.run(arr)

    assert df.shape[0] != 0  # terrible test


def test_all_features_zero_array():
    arr = np.zeros([2, 2000])
    stream = get_example_stream(arr)

    df = stream.run(arr)


def test_all_features_NaN_array():
    arr = np.empty([2, 2000])
    arr[:] = np.nan

    stream = get_example_stream(arr)

    df = stream.run(arr)
