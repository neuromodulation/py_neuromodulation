import py_neuromodulation as nm
import numpy as np


def test_fooof_features(setup_default_stream_fooof):
    default_stream: tuple[np.ndarray, nm.Stream] = setup_default_stream_fooof
    data, stream = default_stream

    generator = nm.stream.RawDataGenerator(
        data,
        stream.sfreq,
        stream.settings.sampling_rate_features_hz,
        stream.settings.segment_length_features_ms,
    )

    _, data_batch = next(generator, None)
    feature_series = stream.data_processor.process(data_batch)
    # since the settings can define searching for "max_n_peaks" peaks
    # there can be None's in the feature_series
    # with a non successful fit, aperiodic features can also be None
    assert feature_series is not None

def test_fooof_zero_values(setup_default_stream_fooof):
    default_stream: tuple[np.ndarray, nm.Stream] = setup_default_stream_fooof
    data, stream = default_stream
    data_zeros = np.zeros_like(data)

    generator = nm.stream.RawDataGenerator(
        data_zeros,
        stream.sfreq,
        stream.settings.sampling_rate_features_hz,
        stream.settings.segment_length_features_ms,
    )

    _, data_batch = next(generator, None)
    feature_series = stream.data_processor.process(data_batch)

    assert all(value is None for value in feature_series.values()), \
        "Expected all feature values to be None for zero input data."

def test_fooof_zero_values_multiple_iterations(setup_default_stream_fooof):
    default_stream: tuple[np.ndarray, nm.Stream] = setup_default_stream_fooof
    data, stream = default_stream
    stream.settings.postprocessing["feature_normalization"] = True
    data_zeros = np.zeros_like(data)

    features = stream.run(data_zeros, out_dir="./test_data",
                          experiment_name="test_fooof_zero_values_multiple_iterations")

    # convert all None values to zero, and check sum of all features except time column
    assert np.nan_to_num(features.iloc[:, :-1]).sum().sum() == 0, \
        "Expected all feature values to be zero for zero input data."