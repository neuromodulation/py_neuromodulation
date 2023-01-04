import os
import sys
import py_neuromodulation as nm
import xgboost
import matplotlib.pyplot as plt
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
)
import pandas as pd


sub = "008"
ses = "EcogLfpMedOff01"
task = "SelfpacedRotationR"
acq = "StimOff"
run = 1
datatype = "ieeg"


# change root directory of the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(''))
if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")

sys.path.append(os.path.dirname(SCRIPT_DIR))

feature_reader = nm_analysis.Feature_Reader(
    feature_dir="/home/timonmerk/Desktop",
    feature_file="sub-008_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-1"
)

feature_reader.feature_arr['ANALOG_R_ROTA_CH'] = (-1*feature_reader.feature_arr['ANALOG_R_ROTA_CH'] > 3e-7) * 1

# plot for a single channel
ch_used = feature_reader.nm_channels.query(
    '(type=="dbs") and (used == 1)'
).iloc[0]["new_name"]

feature_reader.label = feature_reader.feature_arr['ANALOG_R_ROTA_CH']

feature_used = "bandpass_activity"

feature_reader.plot_target_averaged_channel(
    ch=ch_used,
    list_feature_keywords=[feature_used],
    epoch_len=4,   # Length of epoch in seconds
    threshold=0.5, # Threshold to be used for identifying events
)
