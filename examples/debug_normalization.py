import os
import sys
import py_neuromodulation as nm
import xgboost
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_stats
)
from sklearn import metrics, model_selection
import json
import matplotlib.pyplot as plt
import numpy as np
import re

# change root directory of the project
SCRIPT_DIR = os.path.abspath('')  #os.path.dirname()
if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")

sys.path.append(os.path.dirname(SCRIPT_DIR))

sub = "testsub"
ses = "EphysMedOff"
task = "buttonpress"
run = 0
datatype = "ieeg"

# Define run name and access paths in the BIDS format.
RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

PATH_RUN = os.path.join(
    (os.path.join(SCRIPT_DIR, "data")),
    f"sub-{sub}",
    f"ses-{ses}",
    datatype,
    RUN_NAME,
)
PATH_BIDS = os.path.join(SCRIPT_DIR, "data")


(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm_IO.read_BIDS_data(
    PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
)

# Provide a path for the output data. Each re-referencing method has their PATH_OUT
PATH_OUT_ZSCORE_PRE = os.path.join(SCRIPT_DIR, "data", "derivatives", "normalization", "zscore_pre")
nm_channels_zscorePre = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference='default',
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog",),  # We focus only on LFP data
    target_keywords=("SQUARED_ROTATION",),
)

stream_zscorePre = nm.Stream(
    settings=None,
    nm_channels=nm_channels_zscorePre,
    path_grids=None,
    verbose=True,
)

stream_zscorePre.set_settings_fast_compute()


stream_zscorePre.settings['preprocessing']['raw_normalization'] = True
stream_zscorePre.settings['preprocessing']['preprocessing_order'] = ["raw_normalization",]
stream_zscorePre.settings['postprocessing']['feature_normalization'] = True
stream_zscorePre.settings[ "raw_normalization_settings"]["normalization_method"] = {
    "mean": False,
    "median": False,
    "zscore": True,
    "zscore-median": False,
    "quantile": False,
    "power": False,
    "robust": False,
    "minmax": False
}
stream_zscorePre.settings["feature_normalization_settings"]["normalization_method"] = {
    "mean": False,
    "median": False,
    "zscore": False,
    "zscore-median": False,
    "quantile": True,
    "power": False,
    "robust": False,
    "minmax": False
}



stream_zscorePre.init_stream(
    sfreq=sfreq,
    line_noise=line_noise,
    coord_list=coord_list,
    coord_names=coord_names
)

stream_zscorePre.run(
    data=data,
    out_path_root=PATH_OUT_ZSCORE_PRE,
    folder_name=RUN_NAME,
)