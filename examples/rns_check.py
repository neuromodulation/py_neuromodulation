import os
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_stream_offline,
    nm_IO,
    nm_plots,
    nm_stats,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import mne
from scipy import stats
import pandas as pd

# Nexus2/RNS_DataBank/PITT/PIT-RNS1534/iEEG
# PIT-RNS1534_PE20161220-1_EOF_SZ-NZ.EDF


def get_normalized_data(PATH_READ):
    """
    Read edf data, go through each segment and zscore them,
    concatenate normalized epochs.

    (The normalization might not be an optimal approach)
    """

    raw = mne.io.read_raw_edf(PATH_READ)
    data = raw.get_data()

    annots = raw.annotations.description
    onsets = raw.annotations.onset
    time_range_pre = 0
    dat_l = []

    for idx, annot in enumerate(annots):
        if annot == "eof":
            idx_low = raw.times > time_range_pre
            idx_high = raw.times < onsets[idx]
            idx_range = np.where(np.logical_and(idx_low, idx_high) == True)[0]
            dat_l.append(stats.zscore(data[:, idx_range], axis=1))

    data = np.concatenate(dat_l, axis=1)
    return data


if __name__ == "__main__":

    
    PATH_READ = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS0427/iEEG/PIT-RNS0427_PE20181120-1_EOF_SZ-NZ.EDF"
    PATH_OUT = "/home/timonmerk/Documents/PN_OUT"
    sub_name = "PIT-RNS0427"

    data = get_normalized_data(PATH_READ)

    # basic init settings that will initialize the stream
    channels = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 250

    ch_names = list(channels)
    ch_types = ["ecog" for _ in range(len(ch_names))]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference=None,
        bads=None,
        new_names="default",
        used_types=["ecog"],
    )

    stream = nm_stream_offline.Stream(
        settings=None,
        nm_channels=nm_channels,
        verbose=False,  # Change here if you want to see the outputs of the run
    )

    stream.reset_settings()

    # note: I limit here the frequency bands to have reliable data up to 60 Hz
    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [35, 60],
    }

    # INIT Feature Estimation Time Window Length and Frequency
    stream.settings[
        "sampling_rate_features_hz"
    ] = 0.1  # features are estimated every 10s
    stream.settings[
        "segment_length_features_ms"
    ] = 10000  # the duration of 10s is used for feature estimation

    # ENABLE feature types
    stream.settings["features"]["fft"] = True
    stream.settings["features"]["linelength"] = True
    stream.settings["features"]["sharpwave_analysis"] = True
    stream.settings["features"]["mne_connectiviy"] = True

    # SPECIFY specific feature modalities
    stream.settings["mne_connectiviy"]["method"] = "plv"
    stream.settings["mne_connectiviy"]["mode"] = "multitaper"

    stream.settings["fft_settings"]["windowlength_ms"] = 10000
    stream.settings["fft_settings"]["log_transform"] = True
    stream.settings["fft_settings"]["kalman_filter"] = False

    sharpwave_settings_enable = [
        "width",
        "interval",
        "decay_time",
        "rise_time",
        "rise_steepness",
        "decay_steepness",
        "prominence",
        "interval",
        "sharpness",
    ]

    sharpwave_settings_disable = ["peak_left", "peak_right", "trough"]

    for f in sharpwave_settings_disable:
        stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
            f
        ] = False

    for f in sharpwave_settings_enable:
        stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
            f
        ] = True

    # For the distribution of sharpwave features in the interval (e.g. 10s) an estimator need to be defined
    # e.g. mean or max
    stream.settings["sharpwave_analysis_settings"]["estimator"][
        "mean"
    ] = sharpwave_settings_enable

    stream.settings["sharpwave_analysis_settings"]["estimator"]["max"] = [
        "sharpness",
        "prominence",
    ]
    stream.settings["sharpwave_analysis_settings"]["filter_ranges_hz"] = [
        [5, 30],[5, 60]
    ]

    stream.init_stream(
        sfreq=sfreq,
        line_noise=60,
    )

    # data will be saved at the PATH_OUT in folder sub_name
    stream.run(
        data=data,
        folder_name=sub_name,
        out_path_root=PATH_OUT,
    )
