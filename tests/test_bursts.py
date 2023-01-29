import pytest
import numpy as np
from py_neuromodulation import (
    nm_bursts,
    nm_settings,
    nm_stream_offline,
    nm_define_nmchannels,
)


def test_init_wrong_fband():
    settings = nm_settings.get_default_settings()
    settings["burst_settings"]["frequency_bands"] = ["wrong_band"]
    with pytest.raises(Exception):
        nm_bursts.Burst.test_settings(
            settings,
            [
                "ch1",
                "ch2",
            ],
            1000,
        )


def test_init_wrong_treshold():
    settings = nm_settings.get_default_settings()
    settings["burst_settings"]["threshold"] = -1
    with pytest.raises(Exception):
        nm_bursts.Burst.test_settings(
            settings,
            [
                "ch1",
                "ch2",
            ],
            1000,
        )


def test_init_wrong_timeduration():
    settings = nm_settings.get_default_settings()
    settings["burst_settings"]["time_duration_s"] = -1
    with pytest.raises(Exception):
        nm_bursts.Burst.test_settings(
            settings,
            [
                "ch1",
                "ch2",
            ],
            1000,
        )


def test_init_wrong_burst_feature_init():
    settings = nm_settings.get_default_settings()
    settings["burst_settings"]["burst_features"]["duration"] = -1
    with pytest.raises(Exception):
        nm_bursts.Burst.test_settings(
            settings,
            [
                "ch1",
                "ch2",
            ],
            1000,
        )


def test_bursting_duration():
    settings = nm_settings.get_default_settings()
    settings["features"]["bursts"] = True
    settings["postprocessing"]["feature_normalization"] = False
    TIME_DURATION = 10
    sfreq = 1000
    NUM_CH = 1
    time_points_beta = np.arange(0, 1, 1 / sfreq)
    beta_freq = 18

    beta_wave = np.sin(2 * np.pi * beta_freq * time_points_beta)

    ch_names = ["ch0"]

    bursts = nm_bursts.Burst(settings, ch_names, sfreq)

    for _ in range(10):
        data = np.random.random([NUM_CH, 1 * sfreq])
        f = bursts.calc_feature(data, {})

    f_burst = bursts.calc_feature(
        beta_wave + np.random.random([NUM_CH, 1 * sfreq]), {}
    )

    assert (
        f["ch0_bursts_low beta_amplitude_max"]
        < f_burst["ch0_bursts_low beta_amplitude_max"]
    )

    assert (
        f["ch0_bursts_low beta_duration_max"]
        < f_burst["ch0_bursts_low beta_duration_max"]
    )
