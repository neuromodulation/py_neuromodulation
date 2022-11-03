import os
import sys

# change root directory of the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(''))
if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")

sys.path.append(os.path.dirname(SCRIPT_DIR))

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

sub = "008"
ses = "EcogLfpMedOff01"
task = "SelfpacedRotationR"
acq = "StimOff"
run = 1
datatype = "ieeg"

# Define run name and access paths in the BIDS format.
RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_run-{run}"

PATH_RUN = os.path.join(
    "/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data",
    f"sub-{sub}",
    f"ses-{ses}",
    datatype,
    RUN_NAME,
)
PATH_BIDS = "/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data"

# Provide a path for the output data.
PATH_OUT = os.path.join("/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data","derivatives")

feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME
)
feature_reader.feature_arr['ANALOG_R_ROTA_CH'] = (-1*feature_reader.feature_arr['ANALOG_R_ROTA_CH'] > 3e-7) * 1

# plot for a single channel
ch_used = feature_reader.nm_channels.query(
    '(type=="dbs") and (used == 1)'
).iloc[0]["name"]

feature_used = (
    "stft" if feature_reader.settings["features"]["stft"] else "fft"
)

feature_reader.plot_target_averaged_channel(
    ch=ch_used,
    list_feature_keywords=[feature_used],
    epoch_len=4,   # Length of epoch in seconds
    threshold=0.5, # Threshold to be used for identifying events
)