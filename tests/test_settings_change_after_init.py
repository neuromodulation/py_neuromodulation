import numpy as np

import py_neuromodulation as nm


def test_post_init_channels_change():
    """Test if post initialization of channels will also be ported to the feature computation."""

    data = np.random.random((10, 1000))
    fs = 1000

    stream = nm.Stream(
        sfreq=fs, data=data, experiment_name="test_post_init_nm_channels_change"
    )

    # default channel names are "ch{i}"
    # every time the name changes, the "new_name" should also changes
    # this is however only done during initialization
    stream.channels["new_name"] = [f"new_ch_name_{i}" for i in range(10)]

    features = stream.run(out_dir="./test_data")

    assert len([f for f in features.columns if "new_ch_name_0" in f]) != 0


def test_post_init_channels_used_channels_change_single_channel():
    """Test if post initialization of nm_settings will also be ported to the feature computation."""

    np.random.seed(0)
    data = np.random.random((3, 1000))
    sfreq = 1000
    stream = nm.Stream(
        data=data,
        experiment_name="test_post_init_nm_channels_used_channels_change_single_channel",
        sfreq=sfreq,
        sampling_rate_features_hz=11,
    )
    stream.channels["used"] = 0
    stream.channels.loc[1, "used"] = 1

    features = stream.run(out_dir="./test_data")

    chs_not_used = stream.channels[stream.channels["used"] == 0]["new_name"]

    assert (
        np.sum(
            [
                len([c for c in features.columns if c.startswith(ch_not_used)])
                for ch_not_used in chs_not_used
            ]
        )
        == 0
    )


def test_post_init_channels_used_channels_change_multiple_channel():
    """Test if post initialization of nm_settings will also be ported to the feature computation."""

    np.random.seed(0)
    data = np.random.random((3, 1000))
    sfreq = 1000
    stream = nm.Stream(
        data=data,
        experiment_name="test_post_init_nm_channels_used_channels_change_multiple_channel",
        sfreq=sfreq,
        sampling_rate_features_hz=11,
    )
    stream.channels["used"] = 0
    stream.channels.loc[[0, 2], "used"] = 1

    features = stream.run(out_dir="./test_data")

    chs_not_used = stream.channels[stream.channels["used"] == 0]["new_name"]

    assert (
        np.sum(
            [
                len([c for c in features.columns if c.startswith(ch_not_used)])
                for ch_not_used in chs_not_used
            ]
        )
        == 0
    )
