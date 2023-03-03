import numpy as np

from py_neuromodulation import (
    nm_settings,
    nm_stream_offline,
    nm_define_nmchannels,
)

def test_all_features_NaN_values():

    arr = np.random.random([10, 2000])

    settings = nm_settings.get_default_settings()
    settings["features"]["raw_hjorth"] = True
    settings["features"]["return_raw"] = True
    settings["features"]["bandpass_filter"] = True
    settings["features"]["stft"] = True
    settings["features"]["fft"] = True
    settings["features"]["sharpwave_analysis"] = True
    settings["features"]["coherence"] = True
    settings["features"]["fooof"] = True
    settings["features"]["nolds"] = True
    settings["features"]["bursts"] = True
    settings["features"]["linelength"] = True
    settings["features"]["mne_connectivity"] = True

    settings["coherence"]["channels"] = [["ch1", "ch2"]]


    nm_channels = nm_define_nmchannels.get_default_channels_from_data(arr)

    stream = nm_stream_offline.Stream(
        sfreq=1000,
        nm_channels=nm_channels,
        settings=settings,
        verbose=True
    )

    df = stream.run(arr)

    
