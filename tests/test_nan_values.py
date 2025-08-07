import numpy as np

import py_neuromodulation as nm


def test_stream_with_none_data():
    """Test if passing None as the data to a Stream object results in None features."""

    fs = 1000
    data = np.random.random([2, 2000])
    data[0, :] = None

    stream = nm.Stream(sfreq=fs, data=data)

    features = stream.run(
        data, out_dir="./test_data", experiment_name="test_stream_with_none_data"
    )

    # assert if all features if name ch0 are None
    assert len(
        [f for f in features.columns if "ch0" in f and features[f].isna().all()]
    ) == len([f for f in features if "ch0" in f])

    # and check if all features of the second channel are not None
    assert len(
        [f for f in features.columns if "ch1" in f and features[f].notna().all()]
    ) == len([f for f in features if "ch1" in f])

def test_stream_with_nan_and_non_nan_data_in_one_channel():
    """Test if passing a mix of NaN and non-NaN data to a Stream object results in correct features."""

    fs = 1000
    data = np.random.random([2, 3000])
    nan_start = 1000
    nan_end = 2000
    data[0, nan_start:nan_end] = None # Introduce NaNs in the first channel

    stream = nm.Stream(sfreq=fs, data=data)

    features = stream.run(
        data, out_dir="./test_data", experiment_name="test_stream_with_nan_and_non_nan_data"
    )

    idx_nan_features = (features["time"] > nan_start) & (features["time"] < nan_end + stream.settings.segment_length_features_ms)
    idx_not_nan_features = ~idx_nan_features

    features_not_time = features.drop(columns=["time"])
    features_ch0 = features_not_time.filter(like="ch0")

    assert features_ch0[idx_nan_features].isna().all().all(), \
        "Expected all features to be NaN in the time range with NaN data."
    assert features_ch0[idx_not_nan_features].notna().all().all(), \
        "Expected all features to be non-NaN in the time range with non-NaN data."

def test_stream_with_only_nan_data():
    """Test if passing only NaN data to a Stream object results in all features being None."""

    fs = 1000
    data = np.random.random([2, 2000])
    data[:] = None  # Set all data to NaN

    stream = nm.Stream(sfreq=fs, data=data)

    features = stream.run(
        data, out_dir="./test_data", experiment_name="test_stream_with_only_nan_data"
    )

    assert features.filter(like="ch").isna().all().all(), "Expected all features to be NaN."