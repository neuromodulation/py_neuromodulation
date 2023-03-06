from py_neuromodulation import nm_stream_abc, nm_stream_offline, nm_define_nmchannels, nm_settings
import numpy as np


def get_example_stream(test_arr: np.array) -> nm_stream_abc.PNStream:
    settings = nm_settings.get_default_settings()
    settings["features"]["raw_hjorth"] = True
    settings["features"]["return_raw"] = True
    settings["features"]["bandpass_filter"] = True
    settings["features"]["stft"] = True
    settings["features"]["fft"] = True
    settings["features"]["sharpwave_analysis"] = True
    settings["features"]["fooof"] = True
    settings["features"]["bursts"] = True
    settings["features"]["linelength"] = True
    settings["features"]["nolds"] = False
    settings["features"]["mne_connectivity"] = False
    settings["features"]["coherence"] = False

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

    stream = nm_stream_offline.Stream(
        sfreq=1000,
        nm_channels=nm_channels,
        settings=settings,
        verbose=True
    )
    return stream

arr = np.zeros([2, 2000])
stream = get_example_stream(arr)

df = stream.run(arr)

# check if data contains NaN's
assert df.shape[1] == df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).sum()



# demo with random data

arr = np.random.random([2, 2000])

arr = np.empty([2, 2000])
#arr[:] = np.nan

settings = nm_settings.get_default_settings()
settings["features"]["raw_hjorth"] = True
settings["features"]["return_raw"] = True
settings["features"]["bandpass_filter"] = True
settings["features"]["stft"] = True
settings["features"]["fft"] = True
settings["features"]["sharpwave_analysis"] = True
settings["features"]["coherence"] = False
settings["features"]["fooof"] = True
settings["features"]["nolds"] = False
settings["features"]["bursts"] = True
settings["features"]["linelength"] = True
settings["features"]["mne_connectivity"] = False

settings["coherence"]["channels"] = [["ch0", "ch1"]]

settings["segment_length_features_ms"] = 1000
settings["sampling_rate_features_hz"] = 10

nm_channels = nm_define_nmchannels.get_default_channels_from_data(arr)

stream = nm_stream_offline.Stream(
    sfreq=1000,
    nm_channels=nm_channels,
    settings=settings,
    verbose=True
)

df = stream.run(arr)

# could fail

assert df.shape[1] == df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).sum()







arr = np.random.random([10, 5000])

nm_settings.set_settings_fast_compute()

nm_channels = nm_define_nmchannels.get_default_channels_from_data(arr)

settings = nm_settings.get_default_settings()
settings["segment_length_features_ms"] = 2000

stream = nm_stream_offline.Stream(
    sfreq=1000,
    nm_channels=nm_channels,
    settings=settings,
    verbose=True
)

df = stream.run(arr)
print("hallo")
