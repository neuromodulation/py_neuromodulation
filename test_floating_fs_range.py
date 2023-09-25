from sklearn import metrics, model_selection, linear_model
import matplotlib.pyplot as plt

import py_neuromodulation as pn
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
)

import mne
import numpy as np

data = np.random.random((10, 10000))
fs = 1234.417940797

stream = pn.Stream(sfreq=fs, data=data, sampling_rate_features_hz=200)

features = stream.run()

