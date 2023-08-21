import numpy as np
import pytest

from py_neuromodulation import nm_oscillatory, nm_settings


def setup_osc_settings(
    osc_feature_name: str,
    osc_feature_setting: str,
    windowlength_ms: int,
    log_transform: bool,
    kalman_filter: bool,
):

    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings[osc_feature_name] = True
    settings[osc_feature_setting]["windowlength_ms"] = windowlength_ms
    settings[osc_feature_setting]["log_transform"] = log_transform
    settings[osc_feature_setting]["kalman_filter"] = kalman_filter

    return settings

def test_fft_wrong_logtransform_param_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}

    with pytest.raises(Exception) as e_info:
        nm_oscillatory.FFT.test_settings(settings, ch_names, sfreq)


def test_fft_wrong_frequencyband_range_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(Exception):
        nm_oscillatory.FFT.test_settings(settings, ch_names, sfreq)


def test_fft_zero_data():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
        kalman_filter=False,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}
    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)
    fft_obj.test_settings(settings, ch_names, sfreq)

    data = np.ones([len(ch_names), sfreq])
    features_out = fft_obj.calc_feature(data, {})

    for f in features_out.keys():
        assert features_out[f] == 0


def test_fft_random_data():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
        kalman_filter=False,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}
    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)
    fft_obj.test_settings(settings, ch_names, sfreq)

    data = np.random.random([len(ch_names), sfreq])
    features_out = fft_obj.calc_feature(data, {})

    for f in features_out.keys():
        assert features_out[f] != 0


def test_fft_beta_osc():

    ch_names = [
        "ch1",
    ]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
        kalman_filter=False,
    )

    settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)
    fft_obj.test_settings(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = fft_obj.calc_feature(data, {})

    assert (
        features_out["ch1_fft_beta"] > features_out["ch1_fft_theta"]
        and features_out["ch1_fft_beta"] > features_out["ch1_fft_gamma"]
    )

def test_stft_wrong_logtransform_param_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}

    with pytest.raises(Exception) as e_info:
        nm_oscillatory.STFT.test_settings(settings, ch_names, sfreq)


def test_stft_wrong_frequencyband_range_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(Exception):
        nm_oscillatory.STFT.test_settings(settings, ch_names, sfreq)

def test_stft_beta_osc():

    ch_names = [
        "ch1",
    ]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform=False,
        kalman_filter=False,
    )

    settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    stft_obj = nm_oscillatory.STFT(settings, ch_names, sfreq)
    stft_obj.test_settings(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = stft_obj.calc_feature(data, {})

    assert (
        features_out["ch1_stft_beta"] > features_out["ch1_stft_theta"]
        and features_out["ch1_stft_beta"] > features_out["ch1_stft_gamma"]
    )

def test_bp_wrong_logtransform_param_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="bandpass_filter",
        osc_feature_setting="bandpass_filter_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}

    with pytest.raises(Exception) as e_info:
        nm_oscillatory.BandPower.test_settings(settings, ch_names, sfreq)


def test_bp_wrong_frequencyband_range_init():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="bandpass_filter",
        osc_feature_setting="bandpass_filter_settings",
        windowlength_ms=1000,
        log_transform="123",
        kalman_filter=True,
    )
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(Exception):
        nm_oscillatory.BandPower.test_settings(settings, ch_names, sfreq)

def test_bp_non_defined_fband():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="bandpass_filter",
        osc_feature_setting="bandpass_filter_settings",
        windowlength_ms=1000,
        log_transform=True,
        kalman_filter=True,
    )

    settings["frequency_ranges_hz"] = {"theta": [4, 8], "broadband": [10, 600]}
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["theta"] = 1000
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["beta"] = 300

    with pytest.raises(Exception):
        nm_oscillatory.BandPower.test_settings(settings, ch_names, sfreq)

def test_bp_segment_length_fb_exceeds_segment_length_features():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="bandpass_filter",
        osc_feature_setting="bandpass_filter_settings",
        windowlength_ms=1000,
        log_transform=True,
        kalman_filter=True,
    )

    settings["segment_length_features_ms"] = 500
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "broadband": [10, 600]}
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["theta"] = 1000
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["beta"] = 300

    with pytest.raises(Exception):
        nm_oscillatory.BandPower.test_settings(settings, ch_names, sfreq)


def test_bp_zero_data():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings["features"]["bandpass_filter"] = True
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["theta"] = 1000
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["beta"] = 300

    settings["bandpass_filter_settings"]["log_transform"] = False
    settings["bandpass_filter_settings"]["kalman_filter"] = False
    settings["bandpass_filter_settings"]["bandpower_features"]["activity"] = True

    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 20]}
    stft_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)
    stft_obj.test_settings(settings, ch_names, sfreq)

    data = np.zeros([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data, {})

    for f in features_out.keys():
        assert pytest.approx(0, 0.01) == features_out[f]

def test_bp_random_data():

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 30]}
    settings["features"]["bandpass_filter"] = True
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["theta"] = 1000
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["beta"] = 300

    settings["bandpass_filter_settings"]["log_transform"] = False
    settings["bandpass_filter_settings"]["kalman_filter"] = False
    settings["bandpass_filter_settings"]["bandpower_features"]["activity"] = True

    stft_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)
    stft_obj.test_settings(settings, ch_names, sfreq)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data, {})

    for f in features_out.keys():
        assert pytest.approx(0, 0.01) != features_out[f]

def test_bp_beta_osc():

    ch_names = ["ch1",]
    sfreq = 1000

    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings["frequency_ranges_hz"] = {"theta": [4, 8], "beta": [10, 30], "gamma": [50, 60]}

    settings["features"]["bandpass_filter"] = True
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["theta"] = 1000
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["beta"] = 300
    settings["bandpass_filter_settings"]["segment_lengths_ms"]["gamma"] = 100

    settings["bandpass_filter_settings"]["log_transform"] = False
    settings["bandpass_filter_settings"]["kalman_filter"] = False
    settings["bandpass_filter_settings"]["bandpower_features"]["activity"] = True

    bp_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)
    bp_obj.test_settings(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = bp_obj.calc_feature(data, {})

    assert (
        features_out["ch1_bandpass_activity_beta"] > features_out["ch1_bandpass_activity_theta"]
        and features_out["ch1_bandpass_activity_beta"] > features_out["ch1_bandpass_activity_gamma"]
    )
