import py_neuromodulation as nm
import numpy as np


def get_fast_compute_settings():
    settings = nm.nm_settings.get_default_settings()
    settings = nm.nm_settings.reset_settings(settings)
    settings = nm.nm_settings.set_settings_fast_compute(settings)
    settings["preprocessing"] = ["re_referencing", "notch_filter"]
    settings["features"]["fft"] = True
    settings["postprocessing"]["feature_normalization"] = True
    return settings


def get_features(time_end_ms: int, segment_length_features_ms: int):
    data = np.random.random([2, time_end_ms])
    settings = get_fast_compute_settings()
    settings["segment_length_features_ms"] = segment_length_features_ms

    settings["frequency_ranges_hz"] = {
        # "high beta" : [20, 35],
        "low gamma": [60, 80],
        "high gamma": [90, 200],
    }
    stream = nm.Stream(
        sfreq=1000,
        data=data,
        sampling_rate_features_hz=10,
        verbose=False,
        settings=settings,
    )

    features = stream.run(data)
    return features


def test_feature_timing_5s_start_800ms():
    """Test if the features timing duration are computed correctly for a 5s signal with 800ms segment length"""
    time_end_ms = 5000
    segment_length_features_ms = 800
    features = get_features(time_end_ms, segment_length_features_ms)

    assert int(features["time"].iloc[0]) == segment_length_features_ms

    assert int(features["time"].iloc[-1]) == time_end_ms


def test_feature_timing_1s_start_500ms():
    """Test if the features timing duration are computed correctly for a 1s signal with 500ms segment length"""
    time_end_ms = 1000
    segment_length_features_ms = 500
    features = get_features(time_end_ms, segment_length_features_ms)

    assert int(features["time"].iloc[0]) == segment_length_features_ms

    assert int(features["time"].iloc[-1]) == time_end_ms
