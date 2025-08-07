import numpy as np

import py_neuromodulation as nm


def get_example_stream(test_arr: np.ndarray) -> nm.Stream:
    settings = nm.NMSettings.get_default().enable_all_features()
    settings.features.nolds = False
    settings.features.mne_connectivity = False
    settings.features.coherence = False

    channels = nm.utils.get_default_channels_from_data(test_arr)

    stream = nm.Stream(sfreq=1000, channels=channels, settings=settings, verbose=True)
    return stream


def test_all_features_random_array():
    """This test runs's through all enabled features, and check's if they break"""
    np.random.seed(0)
    arr = np.random.random([2, 2000])
    stream = get_example_stream(arr)

    df = stream.run(
        arr, out_dir="./test_data", experiment_name="test_all_features_random_array"
    )

    assert df.shape[0] != 0  # check if not exception was raised


def test_all_features_zero_array():
    arr = np.zeros([2, 2000])

    stream = get_example_stream(arr)
    #stream.settings.features.fooof = False  # Can't use fooof with 0s (log(0) undefined)

    df = stream.run(
        arr, out_dir="./test_data", experiment_name="test_all_features_zero_array"
    )

    # the issue is here that some features, like bandpass activity will be very high
    # negative values,
    # TODO: think about how to check that
    assert df.shape[0] != 0  # check if not exception was raised


def test_all_features_NaN_array():
    arr = np.empty([2, 2000])
    arr[:] = np.nan

    stream = get_example_stream(arr)
    #stream.settings.features.fooof = False  # Can't use fooof nan values

    df = stream.run(
        arr, out_dir="./test_data", experiment_name="test_all_features_NaN_array"
    )

    assert df.shape[0] != 0  # check if not exception was raised
