from pydantic import ValidationError
import pytest
import numpy as np

from py_neuromodulation import nm_oscillatory, NMSettings
from py_neuromodulation.nm_types import FeatureName


def setup_osc_settings(
    osc_feature_name: FeatureName,
    osc_feature_setting: str,
    windowlength_ms: int,
    log_transform: bool,
):
    settings = NMSettings.get_default().reset()

    settings.features[osc_feature_name] = True
    getattr(settings, osc_feature_setting).windowlength_ms = windowlength_ms
    getattr(settings, osc_feature_setting).log_transform = log_transform

    return settings


def setup_bandpass_settings(log_transform: bool):
    settings = NMSettings.get_default().reset()

    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.log_transform = log_transform

    return settings


def test_fft_wrong_logtransform_param_init():
    """Test that settings cannot be initialized with wrong log_transform parameter value"""
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    # TONI: Move with before settings assignment because pydantic will validate the settings
    with pytest.raises(ValidationError):
        settings = setup_osc_settings(
            osc_feature_name="fft",
            osc_feature_setting="fft_settings",
            windowlength_ms=1000,
            log_transform="123",
        )
        # Toni: in order to make this tests work, I had to force validation of the settings
        # in the OscillatoryFeature class __init__ method. This is not ideal because at that
        # point settings would already have been  validate I think, but probably not a big deal in terms of performance

        # TONI: from here down the code won't even be executed...
        settings.frequency_ranges_hz = {"theta": (4, 8), "beta": (10, 20)}
        nm_oscillatory.FFT(settings, ch_names, sfreq)


def test_fft_wrong_frequencyband_range_init():
    """ ???  """
    # TONI: I don't get this test, one should be able to define any frequency range with any name in theory
    # this test fails because FFT does not test the ranges against the bandpass_filter_settings ranges
    # but those are not used by FFT so it should not be teste in the first place
    
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="fft",
        osc_feature_setting="fft_settings",
        windowlength_ms=1000,
        log_transform=False, # TONI: Should not be testing a bad log_transform value here
    )

    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(ValidationError):
        nm_oscillatory.FFT(settings, ch_names, sfreq)


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
    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)

    data = np.ones([len(ch_names), sfreq])
    features_out = fft_obj.calc_feature(data, {})

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
    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)

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
    )

    settings.frequency_ranges_hz = {
        "theta": [4, 8],
        "beta": [10, 28],
        "gamma": [50, 60],
    }

    fft_obj = nm_oscillatory.FFT(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = fft_obj.calc_feature(data, {})

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
        nm_oscillatory.STFT(settings, ch_names, sfreq)


def test_stft_wrong_frequencyband_range_init():
    """ ??? """
    # TONI: same as with FFT, I don't know what is being tested, it seems that
    # it's testing that "broadband" is not in bandpass settings but STFT makes
    # no use of bandpass settings so I don't see the point
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = setup_osc_settings(
        osc_feature_name="stft",
        osc_feature_setting="stft_settings",
        windowlength_ms=1000,
        log_transform=False,
    )
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(ValidationError):
        nm_oscillatory.STFT(settings, ch_names, sfreq)


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

    stft_obj = nm_oscillatory.STFT(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = stft_obj.calc_feature(data, {})

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

    stft_obj = nm_oscillatory.Welch(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = stft_obj.calc_feature(data, {})

    assert (
        features_out["ch1_welch_beta_mean"] > features_out["ch1_welch_theta_mean"]
        and features_out["ch1_welch_beta_mean"] > features_out["ch1_welch_gamma_mean"]
    )


def test_bp_wrong_logtransform_param_init():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.log_transform = "123"

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}

    with pytest.raises(ValidationError):
        nm_oscillatory.BandPower(settings, ch_names, sfreq)


def test_bp_wrong_frequencyband_range_init():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.log_transform = False
    
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    with pytest.raises(ValidationError):
        nm_oscillatory.BandPower(settings, ch_names, sfreq)


def test_bp_non_defined_fband():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.log_transform = False
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}

    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    with pytest.raises(ValidationError):
        nm_oscillatory.BandPower(settings, ch_names, sfreq)


def test_bp_segment_length_fb_exceeds_segment_length_features():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.log_transform = False

    settings.segment_length_features_ms = 500
    settings.frequency_ranges_hz = {"theta": [4, 8], "broadband": [10, 600]}
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    with pytest.raises(Exception):
        nm_oscillatory.BandPower(settings, ch_names, sfreq)


def test_bp_zero_data():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 20]}
    stft_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)

    data = np.zeros([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data, {})

    for f in features_out.keys():
        assert pytest.approx(0, 0.01) == features_out[f]


def test_bp_random_data():
    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 1000

    settings = NMSettings.get_default().reset()

    settings.frequency_ranges_hz = {"theta": [4, 8], "beta": [10, 30]}
    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    stft_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq])
    features_out = stft_obj.calc_feature(data, {})

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

    settings.features["bandpass_filter"] = True
    settings.bandpass_filter_settings.segment_lengths_ms["theta"] = 1000
    settings.bandpass_filter_settings.segment_lengths_ms["beta"] = 300
    settings.bandpass_filter_settings.segment_lengths_ms["gamma"] = 100

    settings.bandpass_filter_settings.log_transform = False
    settings.bandpass_filter_settings.kalman_filter = False
    settings.bandpass_filter_settings.bandpower_features.activity = True

    bp_obj = nm_oscillatory.BandPower(settings, ch_names, sfreq)
    # bp_obj.test_settings(settings, ch_names, sfreq)

    time_duration = 1

    time_points = np.arange(0, time_duration, 1 / sfreq)
    beta_freq = 20

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points)

    np.random.seed(0)
    data = np.random.random([len(ch_names), sfreq]) + beta_wave

    features_out = bp_obj.calc_feature(data, {})

    assert (
        features_out["ch1_bandpass_activity_beta"]
        > features_out["ch1_bandpass_activity_theta"]
        and features_out["ch1_bandpass_activity_beta"]
        > features_out["ch1_bandpass_activity_gamma"]
    )
