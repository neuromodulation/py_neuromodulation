from pydantic import ValidationError
import pytest
import numpy as np

from py_neuromodulation import NMSettings, Stream, features
from py_neuromodulation.utils.types import FEATURE_NAME


def setup_osc_settings(
    osc_feature_name: FEATURE_NAME,
    osc_feature_setting: str,
    windowlength_ms: int,
    log_transform: bool,
):
    settings = NMSettings.get_default().reset()

    settings.features[osc_feature_name] = True
    settings[osc_feature_setting].windowlength_ms = windowlength_ms
    settings[osc_feature_setting].log_transform = log_transform

    return settings


def setup_bandpass_settings(log_transform: bool):
    settings = NMSettings.get_default().reset()

    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.log_transform = log_transform

    return settings


def test_fft_wrong_logtransform_param_init():
    """Test that settings cannot be initialized with wrong log_transform parameter value"""
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    with pytest.raises(ValidationError):
        settings = setup_osc_settings(
            osc_feature_name="fft",
            osc_feature_setting="fft_settings",
            windowlength_ms=1000,
            log_transform="123",
        )
        settings.frequency_ranges_hz = {"theta": (4, 8), "beta": (10, 20)}
        features.FFT(settings, ch_names, sfreq)


def test_fft_frequencyband_range_passing_nyquist_range():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000
    data = np.random.random([len(ch_names), sfreq])

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
    )

    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(AssertionError):
        Stream(sfreq=sfreq, data=data, settings=settings)


def test_fft_zero_data():
    """ """
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
    )
    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}
    fft_obj = features.FFT(settings, ch_names, sfreq)

    data = np.ones([len(ch_names), sfreq])
    features_out = fft_obj.calc_feature(data)

    for f in features_out.keys():
        if "psd_0" not in f:
            assert np.isclose(features_out[f], 0, atol=1e-6)


def test_fft_random_data():
    """Test that FFT feature extraction works with random numbers"""
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False,
    )
    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}
    fft_obj = features.FFT(settings, ch_names, sfreq)

    data = np.random.random([len(ch_names), sfreq])
    features_out = fft_obj.calc_feature(data)

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
    )

    settings.frequency_ranges_hz = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    fft_obj = features.FFT(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = fft_obj.calc_feature(data)

    assert (
        features_out["ch1_fft_beta_mean"] > features_out["ch1_fft_theta_mean"]
        and features_out["ch1_fft_beta_mean"] > features_out["ch1_fft_gamma_mean"]
    )


def test_stft_wrong_logtransform_param_init():
    """ """
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    with pytest.raises(ValidationError):
        settings = setup_osc_settings(
            osc_feature_name="stft",
            osc_feature_setting="stft_settings",
            windowlength_ms=1000,
            log_transform="123",
        )
        features.STFT(settings, ch_names, sfreq)


def test_stft_wrong_frequencyband_range_init():
    """ """
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000
    data = np.random.random([len(ch_names), sfreq])

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform=False,
    )
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(AssertionError):
        Stream(settings=settings, data=data, sfreq=sfreq)


def test_stft_beta_osc():
    ch_names = [
        "ch1",
    ]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform=True,
    )

    settings.frequency_ranges_hz = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    stft_obj = features.STFT(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = stft_obj.calc_feature(data)

    assert (
        features_out["ch1_stft_beta_mean"] > features_out["ch1_stft_theta_mean"]
        and features_out["ch1_stft_beta_mean"] > features_out["ch1_stft_gamma_mean"]
    )


def test_welch_beta_osc():
    ch_names = [
        "ch1",
    ]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="welch",
        osc_feature_setting="welch_settings",
        windowlength_ms=1000,
        log_transform=True,
    )

    settings.frequency_ranges_hz = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    stft_obj = features.Welch(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = stft_obj.calc_feature(data)

    assert (
        features_out["ch1_welch_beta_mean"] > features_out["ch1_welch_theta_mean"]
        and features_out["ch1_welch_beta_mean"] > features_out["ch1_welch_gamma_mean"]
    )


def test_bp_wrong_logtransform_param_init():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.log_transform = "123"

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}

    with pytest.raises(ValidationError):
        features.BandPower(settings, ch_names, sfreq)


def test_bp_wrong_frequencyband_range_init():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.log_transform = False

    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(ValidationError):
        features.BandPower(settings, ch_names, sfreq)


def test_bp_non_defined_fband():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.log_transform = False
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    with pytest.raises(ValidationError):
        features.BandPower(settings, ch_names, sfreq)


def test_bp_segment_length_fb_exceeds_segment_length_features():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.log_transform = False

    settings.segment_length_features_ms = 500
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    with pytest.raises(Exception):
        features.BandPower(settings, ch_names, sfreq)


def test_bp_zero_data():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}
    stft_obj = features.BandPower(settings, ch_names, sfreq)

    data = np.zeros([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data)

    for f in features_out.keys():
        assert pytest.approx(0, 0.01) == features_out[f]


def test_bp_random_data():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 30]}
    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    stft_obj = features.BandPower(settings, ch_names, sfreq)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data)

    for f in features_out.keys():
        assert pytest.approx(0, 0.01) != features_out[f]


def test_bp_beta_osc():
    ch_names = [
        "ch1",
    ]
    sfreq = 1000

    settings = NMSettings.get_default().reset()

    settings.frequency_ranges_hz = {
        "theta": [4, 8],
        "beta": [10, 30],
        "gamma": [50, 60],
    }

    settings.features.bandpass_filter = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300
    settings.bandpass_filter_settings.segment_lengths_ms["gamma"] = 100

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    bp_obj = features.BandPower(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = bp_obj.calc_feature(data)

    assert (
        features_out["ch1_bandpass_activity_beta"]
        > features_out["ch1_bandpass_activity_theta"]
        and features_out["ch1_bandpass_activity_beta"]
        > features_out["ch1_bandpass_activity_gamma"]
    )
