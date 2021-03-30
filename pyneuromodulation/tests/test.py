import json
import os
from os.path import isdir
import sys

from threading import main_thread
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from scipy.optimize.optimize import main

import mne_bids

# get py_neuromoulation files
PATH_PYNEUROMODULATION = r'C:\Users\ICN_admin\Documents\py_neuromodulation'
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
import define_M1
import generator
import rereference
import settings


def read_example_data(PATH_PYNEUROMODULATION):
    """This test function return a data batch and automatic initialized M1 datafram

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing 
    """
    # read examplary settings file
    with open(os.path.join(PATH_PYNEUROMODULATION, 'examples', 'settings.json'), \
                    encoding='utf-8') as json_file:
        settings = json.load(json_file)

    # define BIDS Example Path and read example data 
    BIDS_EXAMPLE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation', \
                    'tests', 'data')
    PATH_RUN = os.path.join(BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff', \
                    'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")
    PATH_M1 = None 
    entities = mne_bids.get_entities_from_fname(PATH_RUN)
    bids_path = mne_bids.BIDSPath(subject=entities['subject'], \
                    session=entities['session'], task=entities["task"], \
        run=entities["run"], acquisition=entities['acquisition'], datatype='ieeg', \
                    root=BIDS_EXAMPLE_PATH)
    raw_arr = mne_bids.read_raw_bids(bids_path)
    ieeg_raw = raw_arr.get_data()
    fs = int(np.ceil(raw_arr.info['sfreq']))

    # define M1 default file 
    df_M1 = pd.read_csv(PATH_M1, sep="\t") if PATH_M1 is not None and os.path.isfile(PATH_M1) \
        else define_M1.set_M1(raw_arr.ch_names, raw_arr.get_channel_types())

    gen = generator.ieeg_raw_generator(ieeg_raw, settings, fs) 

    ieeg_batch = next(gen, None)

    return ieeg_batch, df_M1

def initialize_rereference(df_M1):
    """The rereference class get's here instantiated given the supplied df_M1 table

    Args:
        df_M1 (pd Dataframe): rereference specifying table  

    Returns:
        RT_rereference: Rereference object
    """
    # define rereference attributes
    ch_names = list(df_M1['name'])
    refs = df_M1['rereference']
    to_ref_idx = np.array(df_M1[(df_M1['target'] == 0) & (df_M1['used'] == 1) & \
                    (df_M1["rereference"] != "None")].index)

    to_ref_idx = np.array(df_M1[(df_M1['used'] == 1)].index)

    cortex_idx = np.where(df_M1.ECOG == 1)[0]
    subcortex_idx = np.array(df_M1[(df_M1["ECOG"] == 0) & \
                    (df_M1['used'] == 1) & (df_M1['target'] == 0)].index)

    ref_here = rereference.RT_rereference(ch_names, refs, to_ref_idx,\
                    cortex_idx, subcortex_idx, split_data=False)
    return ref_here

def test_rereference(ref_here, ieeg_batch, df_M1):
    """
    Args:
        ref_here (RT_rereference): Rereference initialized object
        ieeg_batch (np.ndarray): sample data 
        df_M1 (pd.Dataframe): rereferencing dataframe
    
    """
    ref_dat = ref_here.rereference(ieeg_batch)

    print("test the channels which are used but not rereferenced") 
    for no_ref_idx in np.where((df_M1.rereference == "None") & df_M1.used == 1)[0]:
        assert_array_equal(ref_dat[no_ref_idx,:], ieeg_batch[no_ref_idx,:])

    print("test ecog average channels")
    for ecog_ch_idx in np.where((df_M1.ECOG == 1) & (df_M1.rereference == 'average'))[0]:
        assert_array_equal(ref_dat[ecog_ch_idx,:], ieeg_batch[ecog_ch_idx,:] - \
                ieeg_batch[(df_M1["ECOG"] == 1) & (df_M1.index != ecog_ch_idx)].mean(axis=0))

    print("test bipolar rereferenced channels")
    for bp_reref_idx in [ch_idx for ch_idx, ch in \
                        enumerate(df_M1.rereference) if ch in list(df_M1.name)]:
        # bp_reref_idx is the channel index of the rereference anode 
        # referenced_bp_channel is the channel index which is the rereference cathode 
        referenced_bp_channel = np.where(df_M1.iloc[bp_reref_idx]['rereference'] == df_M1.name)[0][0]
        assert_array_equal(ref_dat[bp_reref_idx,:], \
                        ieeg_batch[bp_reref_idx,:] - ieeg_batch[referenced_bp_channel,:])


if __name__ == "__main__":

    ieeg_batch, df_M1 = read_example_data(PATH_PYNEUROMODULATION)
    ref_here = initialize_rereference(df_M1)
    test_rereference(ref_here, ieeg_batch, df_M1)
    settings.test_settings(os.path.join(PATH_PYNEUROMODULATION, 'examples',
                                        'settings.json'))