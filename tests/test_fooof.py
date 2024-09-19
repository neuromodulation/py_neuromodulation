import py_neuromodulation as nm
import numpy as np


def test_fooof_features(setup_default_stream_fast_compute):
    default_stream: tuple[np.ndarray, nm.Stream] = setup_default_stream_fast_compute
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
