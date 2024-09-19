import numpy as np
import pytest

import py_neuromodulation as nm


def test_stream_init():
    """Test if stream initialization with passed data will setup channels correctly"""
    np.random.seed(0)
    data = np.random.random((10, 1000))
    sfreq = 1000
    stream = nm.Stream(sfreq=sfreq, data=data, sampling_rate_features_hz=11)

    assert stream.channels.shape[0] == 10
    assert stream.settings.sampling_rate_features_hz == 11


def test_stream_init_no_sfreq():
    """Check if stream initialization without sfreq will raise an error"""
    np.random.seed(0)
    data = np.random.random((10, 1000))
    with pytest.raises(Exception):
        nm.Stream(data=data, sampling_rate_features_hz=11)


def test_init_warning_no_used_channel():
    """Check if a warning is raised when a stream is initialized with channels, but no row has used == 1 and target == 0"""
    np.random.seed(0)
    data = np.random.random((10, 1000))
    sfreq = 1000
    stream = nm.Stream(sfreq=sfreq, data=data, sampling_rate_features_hz=11)
    channels = stream.channels
    channels["used"] = 0

    with pytest.raises(Exception):
        nm.Stream(
            sfreq=sfreq,
            data=data,
            channels=channels,
            sampling_rate_features_hz=11,
        )
