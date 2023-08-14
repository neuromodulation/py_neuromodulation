import numpy as np

import py_neuromodulation as nm


def test_stream_init():
    """Test if stream initialization with passed data will setup nm_channels correctly
    """
    data = np.random.random((10, 1000))
    sfreq = 100
    stream = nm.Stream(sfreq=sfreq, data=data, sampling_rate_features_hz=11)

    assert stream.nm_channels.shape[0] == 10
    assert stream.settings["sampling_rate_features_hz"] == 11