import py_neuromodulation as nm
import mne.io
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

## Check data with mne.io.read_raw_brainvision

### General data locations
GEN_DATA_LOC = 'E:/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - Data'
BERLIN_DATASET = '/BIDS_01_Berlin_Neurophys/rawdata'
BEIJING_DATASET = '/BIDS_Beijing_ECOG_LFP/rawdata'
PITT_DATASET = '/BIDS_Pittsburgh_Gripforce/rawdata'
WASHINGTON_DATASET = np.nan

### Specifics (try to arrange this such that it can be done in a loop)

run_files_beijing = BIDSLayout(GEN_DATA_LOC+BEIJING_DATASET)

# Separate for Berlin dataset
# subjectfolders = [folder for folder in os.listdir(GEN_DATA_LOC+BERLIN_DATASET+'/rawdata') if 'sub-E' in folder]
# Standard
#loclist = [] # List containing: subject folder name, trial folder names and corresponding RUN_NAME
# Example for BEIJING_DATASET
#for subfol in os.listdir(GEN_DATA_LOC+BEIJING_DATASET+'/rawdata'):
#    if 'sub' in subfol:
#        for trialfol in os.listdir(GEN_DATA_LOC+BEIJING_DATASET+'/rawdata/'+subfol):
#            if 'Ephys' in trialfol:
#                for datafiles in os.listdir(GEN_DATA_LOC+BEIJING_DATASET+'/rawdata/'+subfol+'/'+trialfol+'/'+'ieeg'):
#                    if 'ieeg' in datafiles:
#                        runname = datafiles[:-9]
#                        loclist.append([subfol, trialfol, runname])
#                        break

PATH_BIDS = GEN_DATA_LOC+BEIJING_DATASET
layout = BIDSLayout(PATH_BIDS)
getfiles = layout.get(task=['ButtonPressL','ButtonPressR'], extension='.vhdr')
runpathlist = [f.path for f in getfiles]


### Do 1 feature extraction
for i in range(1): # Change to for i in loclist
    PATH_RUN = runpathlist[i][:-5] # Remove later

    #RUN_NAME = data[2]
    #PATH_RUN = GEN_DATA_LOC+BEIJING_DATASET+'/rawdata/'+data[0]+'/'+data[1]+'/ieeg/'+RUN_NAME
    PATH_BIDS = GEN_DATA_LOC+BEIJING_DATASET
    PATH_OUT = 'D:/Glenn/Features_out/BEIJING'
    datatype = 'ieeg'

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

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=["MOV_RIGHT_CLEAN","MOV_LEFT_CLEAN"]
    )

    nm_channels


