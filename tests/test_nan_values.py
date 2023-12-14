import numpy as np

import py_neuromodulation as nm


def test_stream_with_none_data():
    """Test if passing None as the data to a Stream object results in None features."""

    fs = 1000
    data = np.random.random([2, 2000])
    data[0, :] = None

    stream = nm.Stream(fs, data)

    features = stream.run(data)

    # assert if all features if name ch0 are None
    assert len(
        [f for f in features.columns if "ch0" in f and features[f].isna().all()]
    ) == len([f for f in features if "ch0" in f])

    # and check if all features of the second channel are not None
    assert len(
        [
            f
            for f in features.columns
            if "ch1" in f and features[f].notna().all()
        ]
    ) == len([f for f in features if "ch1" in f])
