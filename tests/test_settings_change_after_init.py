import numpy as np

import py_neuromodulation as pn
from py_neuromodulation import nm_settings


def test_post_init_nm_channels_change():
    """Test if post initialization of nm_channels will also be ported to the feature computation.
    """

    data = np.random.random((10, 1000))
    fs = 1000

    stream = pn.Stream(fs, data)

    # default channel names are "ch{i}"
    # every time the name changes, the "new_name" should also changes
    # this is however only done during initialization
    stream.nm_channels["new_name"] = [f"new_ch_name_{i}" for i in range(10)]

    features = stream.run()

    assert len([f for f in features.columns if "new_ch_name_0" in f]) != 0


def test_post_init_settings_change():
    """Test if post initialization of nm_settings will also be ported to the feature computation.
    """

    data = np.random.random((10, 1000))
    fs = 1000

    settings = nm_settings.get_default_settings()

    settings["features"]["fft"] = True

    stream = pn.Stream(fs, data)

    stream.settings["features"]["fft"] = False

    features = stream.run()

    assert len([f for f in features.columns if "fft" in f]) == 0

