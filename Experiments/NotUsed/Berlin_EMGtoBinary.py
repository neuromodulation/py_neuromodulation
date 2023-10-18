import py_neuromodulation as nm
import mne
from bids import BIDSLayout
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
    nm_stats
)
from sklearn import (
    metrics,
    model_selection,
)

import xgboost
import matplotlib.pyplot as plt
import numpy as np
import os

## Start by visualisation of the data

RAW = mne.io.read_raw_brainvision(r'C:\code\glenn_pynm\examples\data\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-3_ieeg.vhdr', preload=True)
RAW.notch_filter(50)
mne.viz.plot_raw(RAW)
