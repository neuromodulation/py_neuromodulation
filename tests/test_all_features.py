import numpy as np

import py_neuromodulation as nm


def get_example_stream(
    test_arr: np.ndarray, experiment_name: str = "test"
) -> nm.Stream:
    settings = nm.NMSettings.get_default().enable_all_features()
    settings.features.nolds = False
    settings.features.mne_connectivity = False
    settings.features.coherence = False

    channels = nm.utils.create_default_channels_from_data(test_arr)

    stream = nm.Stream(
        data=test_arr,
        sfreq=1000,
        channels=channels,
        settings=settings,
        verbose=True,
        experiment_name=experiment_name,
    )
    return stream


def test_all_features_random_array():
    """This test runs's through all enabled features, and check's if they break"""
    np.random.seed(0)
    arr = np.random.random([2, 2000])
    stream = get_example_stream(arr, experiment_name="test_all_features_random_array")

    df = stream.run(out_dir="./test_data")

    assert df.shape[0] != 0  # terrible test


def test_all_features_zero_array():
    arr = np.zeros([2, 2000])

    stream = get_example_stream(arr, experiment_name="test_all_features_zero_array")
    stream.settings.features.fooof = False  # Can't use fooof with 0s (log(0) undefined)

    df = stream.run(out_dir="./test_data")

    assert df.shape[0] != 0  # terrible test


def test_all_features_NaN_array():
    arr = np.empty([2, 2000])
    arr[:] = np.nan

    stream = get_example_stream(arr, experiment_name="test_all_features_NaN_array")
    stream.settings.features.fooof = False  # Can't use fooof nan values

    df = stream.run(out_dir="./test_data")

    assert df.shape[0] != 0  # terrible test
