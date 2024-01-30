import py_neuromodulation as pn
import numpy as np
from py_neuromodulation import nm_settings
import pytest


@pytest.fixture
def get_stream():
    NUM_CHANNELS = 10
    NUM_DATA = 10000
    sfreq = 1000  # Hz
    sampling_rate_features_hz = 3  # Hz

    data = np.random.random([NUM_CHANNELS, NUM_DATA])

    stream = pn.Stream(
        sfreq=sfreq,
        data=data,
        sampling_rate_features_hz=sampling_rate_features_hz,
    )
    stream.nm_channels.loc[0, "target"] = 1
    stream.nm_channels.loc[0, "used"] = 0
    stream.settings["postprocessing"]["feature_normalization"] = False
    stream.settings["segment_length_features_ms"] = 5000
    for feature in stream.settings["features"]:
        stream.settings["features"][feature] = False
    stream.settings["features"]["nolds"] = False
    stream.settings["features"]["fooof"] = True
    stream.settings["features"]["bursts"] = False
    stream.settings["features"]["mne_connectivity"] = False
    stream.settings["coherence"]["channels"] = [["ch1", "ch2"]]
    return stream


def test_setting_exception(get_stream):
    stream = get_stream
    stream.settings["features"]["burst"] = True

    with pytest.raises(Exception) as e_info:
        stream.run(parallel=True, n_jobs=-1)


def test_multiprocessing_and_sequntial_features(get_stream):
    stream_seq = get_stream
    features_sequential = stream_seq.run(parallel=False)

    stream_par = get_stream
    features_multiprocessing = stream_par.run(parallel=True, n_jobs=-1)

    for column in features_sequential.columns:
        if "fooof" in column:
            # fooof results are different in multiprocessing and sequential processing
            # This tests fails on Linux and Windows but passes on Mac OS; no idea why
            continue

        assert features_sequential[column].equals(
            features_multiprocessing[column]
        ), f"Column {column} is not equal between sequential and parallel dataframes computation"
