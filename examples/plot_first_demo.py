"""
First Demo
==========

This Demo will showcase the feature estimation and
examplar analysis using simulated data.
"""

import py_neuromodulation as py_nm

from py_neuromodulation import (
    nm_analysis,
    nm_define_nmchannels,
    nm_plots
    
)
import numpy as np
from matplotlib import pyplot as plt

# %%
# Data Simulation
# ---------------
# We will now generate some exemplar data with 5 second duration for 6 channels with a sample rate of 1 kHz. 

def generate_random_walk(NUM_CHANNELS, TIME_DATA_SAMPLES):
    # from https://towardsdatascience.com/random-walks-with-python-8420981bc4bc
    dims = NUM_CHANNELS
    step_n = TIME_DATA_SAMPLES-1
    step_set = [-1, 0, 1]
    origin = (np.random.random([1,dims])-0.5)*1 # Simulate steps in 1D
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    return path.T

NUM_CHANNELS = 6
sfreq = 1000
TIME_DATA_SAMPLES = 5 * sfreq
data = generate_random_walk(NUM_CHANNELS, TIME_DATA_SAMPLES)
time = np.arange(0, TIME_DATA_SAMPLES/sfreq, 1/sfreq)

plt.figure(figsize=(8,4), dpi=100)
for ch_idx in range(data.shape[0]):
    plt.plot(time, data[ch_idx, :])
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Example random walk data")

# %%
# Now let’s define the necessary setup files will be use for data
# preprocessing and feature estimation. Py_neuromodualtion is based on two
# parametrization files: the *nm_channels.tsv* and the *nm_setting.json*.
# 
# nm_channels
# ~~~~~~~~~~~
# 
# The *nm_channel* dataframe. This dataframe contains the columns
# 
# +-----------------------------------+-----------------------------------+
# | Column name                       | Description                       |
# +===================================+===================================+
# | **name**                          | name of the channel               |
# +-----------------------------------+-----------------------------------+
# | **rereference**                   | different channel name for        |
# |                                   | bipolar rereferencing, or         |
# |                                   | avereage for commono average      |
# |                                   | rereferencing                     |
# +-----------------------------------+-----------------------------------+
# | **used**                          | 0 or 1, channel selection         |
# +-----------------------------------+-----------------------------------+
# | **target**                        | 0 or 1, for some decoding         |
# |                                   | applications we can define target |
# |                                   | chanenls, e.g. EMG channels       |
# +-----------------------------------+-----------------------------------+
# | **type**                          | channel type according to the     |
# |                                   | `mne-python`_ toolbox             |
# |                                   |                                   |
# |                                   |                                   |
# |                                   |                                   |
# |                                   |                                   |
# |                                   | e.g. ecog, eeg, ecg, emg, dbs,    |
# |                                   | seeg etc.                         |
# +-----------------------------------+-----------------------------------+
# | **status**                        | good or bad, used for channel     |
# |                                   | quality indication                |
# +-----------------------------------+-----------------------------------+
# | **new_name**                      | this keyword can be specified to  |
# |                                   | indicate for example the used     |
# |                                   | rereferncing scheme               |
# +-----------------------------------+-----------------------------------+
# 
# .. _mne-python: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html#sphx-glr-auto-tutorials-raw-10-raw-overview-py
# The nm_stream can either created as a *.tsv* text file, or as a pandas
# dataframe. There are some helper function that let you create the
# nm_channels without much effort:

nm_channels = nm_define_nmchannels.get_default_channels_from_data(data, car_rereferencing=True)

nm_channels

# %% Using this function default channel names and a common average rereference scheme is specified. Alternatively the *nm_define_nmchannels.set_channels* function can be used to pass each column values.
# nm_settings
# -----------
# Next, we will initialize the nm_settings dictionary and use the default settings, reset them, and enable a subset of features:

settings = py_nm.nm_settings.get_default_settings()
settings = py_nm.nm_settings.reset_settings(settings)


# %%
# The settings itself is a .json file which contains the parametrization for processing, feature estimation, postprocessing and definition which which sampling sampling rate features are being calculated. In this example 'sampling_rate_features_hz' is specified to be 10 Hz, so every 100ms a new set of features is calculated.
# 
# For many features the 'segment_length_features_ms' specifies the time dimension of the raw signal being used for feature calculation. Here it is specified to be 1000 ms.
# 
# We will now enable the features:
# 
# * fft
# * bursts
# * sharpwave
# 
# and stay with the default preprcessing methods:
# 
# * notch_filter
# * re_referencing
# 
# and use *z-score* postprocessing normalization.

settings["features"]["fft"] = True
settings["features"]["bursts"] = True
settings["features"]["sharpwave_analysis"] = True

# %%
# We are now ready to go to instantiate the *Stream* and call the *run* method for feature estimation:

stream = py_nm.Stream(
    settings=settings,
    nm_channels=nm_channels,
    verbose=True,
    sfreq=sfreq,
    line_noise=50
)

features = stream.run(data)

# %%
# Feature Analysis
# ----------------

# Ok, so there is a lot of output, which we could omit by verbose beinng False, but let's have a look what was being computed. We will therefore use the nm_analysis class to showcase some functions. For multi-run or subject analyze we will pass here the feature_file "sub" default directory:

analyzer = nm_analysis.Feature_Reader(
    feature_dir=stream.PATH_OUT,
    feature_file=stream.PATH_OUT_folder_name
)

# %% 
# Let's have a look at the resulting dataframe, lying in the "feature_arr" dataframe: 

analyzer.feature_arr.iloc[:10, :7]

# %%
# Seems like a lot of features were calculated. The 'time' columns tells us about each row time index. For the 6 specified channels, it is each 31 features. We can now use some in-built plotting functions for visualization.
# 
# Note: Due to the simulation data, some of the features have constant values, which are not displayed throught the image normalization.

analyzer.plot_all_features(ch_used="ch1")

# %%
nm_plots.plot_corr_matrix(
    figsize=(25,25),
    show_plot=True,
    feature=analyzer.feature_arr,
)

# %%
# The upper correlation matrix shows the correlation of every feature of every channel to every other.
# This notebook demonstrated a first demo how features can quickly be generated. For further feature modalities and decoding applications check out the next notebooks.

