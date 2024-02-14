
from py_neuromodulation.nm_filter_preprocessing import PreprocessingFilter

def test_preprocessing_filter(setup):
    ch_names, ch_types, bads, data_batch = setup

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    preprocessing_filter = PreprocessingFilter(settings, sfreq)

    data_filtered = preprocessing_filter.process(data_batch)

    assert data_filtered.shape == data_batch.shape

    assert data_filtered != data_batch