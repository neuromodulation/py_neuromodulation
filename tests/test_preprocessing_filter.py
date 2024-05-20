import numpy as np
from scipy import signal

from py_neuromodulation import NMSettings
from py_neuromodulation.nm_filter_preprocessing import PreprocessingFilter


def test_preprocessing_within_pipeline(setup_default_stream_fast_compute):
    data, stream = setup_default_stream_fast_compute

    stream.settings.preprocessing.append("preprocessing_filter")

    stream.settings.preprocessing_filter.bandstop_filter = True
    stream.settings.preprocessing_filter.bandpass_filter = True
    stream.settings.preprocessing_filter.lowpass_filter = True
    stream.settings.preprocessing_filter.highpass_filter = True

    stream.sfreq

    try:
        _ = stream.run(data[:, : int(stream.sfreq * 2)])
    except Exception as e:
        assert False, f"Error in pipeline including preprocess filtering : {e}"


def test_preprocessing_filter_lowpass():
    data_batch = np.random.random([1, 1000])

    settings = NMSettings.get_default()
    settings.preprocessing.append("preprocessing_filter")
    settings.preprocessing_filter.lowpass_filter = True
    settings.preprocessing_filter.highpass_filter = False
    settings.preprocessing_filter.bandpass_filter = False
    settings.preprocessing_filter.bandstop_filter = False

    settings.preprocessing_filter.lowpass_filter_settings.frequency_cutoff_hz = 100

    sfreq = 1000

    preprocessing_filter = PreprocessingFilter(settings, sfreq)
    data_filtered = preprocessing_filter.process(data_batch)

    # compute a scipy signal welch to check if the filter worked
    f, Pxx = signal.welch(data_batch, fs=sfreq, nperseg=1000)
    f, Pxx_f = signal.welch(data_filtered, fs=sfreq, nperseg=1000)

    # check if the power in the frequency range of the lowpass filter is reduced
    assert np.mean(Pxx_f[0, 100:500]) < np.mean(Pxx[0, 100:500])


def test_preprocessing_filter_highpass():
    data_batch = np.random.random([1, 1000])

    settings = NMSettings.get_default()
    settings.preprocessing.append("preprocessing_filter")
    settings.preprocessing_filter.highpass_filter = True
    settings.preprocessing_filter.lowpass_filter = False
    settings.preprocessing_filter.bandpass_filter = False
    settings.preprocessing_filter.bandstop_filter = False

    settings.preprocessing_filter.highpass_filter_settings.frequency_cutoff_hz = 100

    sfreq = 1000

    preprocessing_filter = PreprocessingFilter(settings, sfreq)
    data_filtered = preprocessing_filter.process(data_batch)

    # compute a scipy signal welch to check if the filter worked
    f, Pxx = signal.welch(data_batch, fs=sfreq, nperseg=1000)
    f, Pxx_f = signal.welch(data_filtered, fs=sfreq, nperseg=1000)

    # check if the power in the frequency range of the highpass filter is reduced
    assert np.mean(Pxx_f[0, 0:100]) < np.mean(Pxx[0, 0:100])


def test_preprocessing_filter_bandstop():
    data_batch = np.random.random([1, 1000])

    settings = NMSettings.get_default()
    settings.preprocessing.append("preprocessing_filter")
    settings.preprocessing_filter.bandstop_filter = True
    settings.preprocessing_filter.bandpass_filter = False
    settings.preprocessing_filter.lowpass_filter = False
    settings.preprocessing_filter.highpass_filter = False

    settings.preprocessing_filter.bandstop_filter_settings.frequency_low_hz = 100
    settings.preprocessing_filter.bandstop_filter_settings.frequency_high_hz = 160

    sfreq = 1000

    preprocessing_filter = PreprocessingFilter(settings, sfreq)
    data_filtered = preprocessing_filter.process(data_batch)

    # compute a scipy signal welch to check if the filter worked
    f, Pxx = signal.welch(data_batch, fs=sfreq, nperseg=1000)
    f, Pxx_f = signal.welch(data_filtered, fs=sfreq, nperseg=1000)

    # check if the power in the frequency range of the bandstop filter is reduced
    assert np.mean(Pxx_f[0, 100:160]) < np.mean(Pxx[0, 100:160])


def test_preprocessing_filter_bandpass():
    data_batch = np.random.random([1, 1000])

    settings = NMSettings.get_default()
    settings.preprocessing.append("preprocessing_filter")
    settings.preprocessing_filter.bandstop_filter = False
    settings.preprocessing_filter.bandpass_filter = True
    settings.preprocessing_filter.lowpass_filter = False
    settings.preprocessing_filter.highpass_filter = False

    settings.preprocessing_filter.bandpass_filter_settings.frequency_low_hz = 100
    settings.preprocessing_filter.bandpass_filter_settings.frequency_high_hz = 160

    sfreq = 1000

    preprocessing_filter = PreprocessingFilter(settings, sfreq)
    data_filtered = preprocessing_filter.process(data_batch)

    # compute a scipy signal welch to check if the filter worked
    f, Pxx = signal.welch(data_batch, fs=sfreq, nperseg=1000)
    f, Pxx_f = signal.welch(data_filtered, fs=sfreq, nperseg=1000)

    # check if the power in the frequency range of the bandpass filter is reduced
    assert np.mean(Pxx_f[0, 0:100]) < np.mean(Pxx[0, 0:100])
    assert np.mean(Pxx_f[0, 160:500]) < np.mean(Pxx[0, 160:500])
