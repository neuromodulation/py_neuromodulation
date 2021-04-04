import json
import multiprocessing
import os
import scipy.io as spio

import sys
sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')

import numpy as np
import pandas as pd

from bids import BIDSLayout
import mne_bids

import define_M1
import features
import generator
import rereference
import resample
import run_analysis
from settings import test_settings

PATH_LFP = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\burst identification\tLfpOffL12"
PATH_BURST = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\burst identification\burstLfpOffL12"

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



# read file 
fs_label = 200  # Hz
burstLfpOffL12 = loadmat(PATH_BURST)
label = burstLfpOffL12["burstLfpOffL12"]["trial"]

fs = 1000  # Hz
line_noise = 50 
tLfpOffL12 = loadmat(PATH_LFP)
# expand dimension s.t. generator (ch, time) won't fail
data = np.expand_dims(tLfpOffL12["tLfpOffL12"]["trial"], axis=0)

# read settings 
with open('settings.json', encoding='utf-8') as json_file:
    settings = json.load(json_file)

test_settings(settings, verbose=True)

# create M1 
PATH_M1 = None
ch_names = ["STN_LEFT_12"]
ch_types = ["lfp"]
df_M1 = pd.read_csv(PATH_M1, sep="\t") \
    if PATH_M1 is not None and os.path.isfile(PATH_M1) \
    else define_M1.set_M1(ch_names, ch_types)
df_M1["used"] = 1

ch_names = list(df_M1['name'])
refs = df_M1['rereference']

to_ref_idx = np.array(df_M1[(df_M1['used'] == 1)].index)

cortex_idx = np.where(df_M1.type == 'ecog')[0]
subcortex_idx = np.array(df_M1[(df_M1["type"] == 'seeg') | (df_M1['type'] == 'dbs')
                                | (df_M1['type'] == 'lfp')].index)

# create generator 
gen = generator.ieeg_raw_generator(data, settings, fs)

feature_idx = np.where(np.logical_and(np.array((df_M1["used"] == 1)),
                                          np.array((df_M1["target"] == 0))))[0]
used_chs = np.array(ch_names)[feature_idx].tolist()

# initialize features 
fs_new = fs
features_ = features.Features(s=settings, fs=fs_new, line_noise=line_noise,
                              channels=used_chs)

# call now run_analysis.py
df_ = run_analysis.run(gen=gen, features=features_, settings=settings, 
                       ref_here=None, used=feature_idx,
                       resample_=None)

print(df_.shape)                     

folder_name = "burst_LFP_L_12"

if not os.path.exists(os.path.join(settings["out_path"], folder_name)):
        os.makedirs(os.path.join(settings["out_path"], folder_name))
df_.to_csv(os.path.join(settings["out_path"], folder_name,
                               folder_name+"_FEATURES.csv"))