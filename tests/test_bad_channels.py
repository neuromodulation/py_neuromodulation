# create a test with two channels, one bad, check then if 
# features are only calculated for the good channel

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
import py_neuromodulation as nm

def test_bad_channels_feature_extraction(setup_default_stream_fast_compute):
    default_stream: tuple[np.ndarray, nm.Stream] = setup_default_stream_fast_compute
    data, stream = default_stream

    # make ecog channels status bad, else good
    stream.channels["status"] = ["bad" if ch == "ecog" else "good" for ch in stream.channels["type"]]

    features = stream.run(data, out_dir="./test_data",
              experiment_name="test_bad_channels_feature_extraction")

    # check if features were only computed for channels starting with LFP, no ECoG
    assert all([f.startswith("LFP") for f in list(features.columns) if f != "time"]), \
        "Features were computed for non-DBS channels, which should not happen."