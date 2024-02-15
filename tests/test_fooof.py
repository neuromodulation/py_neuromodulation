from py_neuromodulation import nm_generator


def test_fooof_features(setup_default_stream_fast_compute):

    data, stream = setup_default_stream_fast_compute

    generator = nm_generator.raw_data_generator(
        data, stream.settings, stream.sfreq
    )
    data_batch = next(generator, None)
    feature_series = stream.run_analysis.process(data_batch)
    # since the settings can define searching for "max_n_peaks" peaks
    # there can be None's in the feature_series
    # with a non successful fit, aperiodic features can also be None
    assert feature_series is not None
