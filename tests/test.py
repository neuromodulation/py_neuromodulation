import json
import os
import sys
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import mne_bids
from pathlib import Path

# first parent to get test folder, second pyneuromodulation folder, then py_neuromodulation
PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'examples'))

import define_M1
import generator
import rereference
import settings
import sharpwaves
import IO

import example_BIDS
import example_ML
import example_read_features

def read_example_data(PATH_PYNEUROMODULATION):
    """This test function return a data batch and automatic initialized M1 datafram

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing
        settings_wrapper (settings.py): settings.json
        fs (float): example sampling frequency
    """

    # read and test settings first to obtain BIDS path
    settings_wrapper = settings.SettingsWrapper(
        settings_path=os.path.join(PATH_PYNEUROMODULATION, 'examples',
                                   'settings.json'))

    BIDS_EXAMPLE_PATH = os.path.join(
        PATH_PYNEUROMODULATION, 'pyneuromodulation', 'tests', 'data')

    settings_wrapper.settings['BIDS_path'] = BIDS_EXAMPLE_PATH
    settings_wrapper.settings['out_path'] = os.path.join(
        BIDS_EXAMPLE_PATH, 'derivatives')

    PATH_RUN = os.path.join(BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff',
                            'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")
    # read BIDS data
    raw_arr, raw_arr_data, fs, line_noise = IO.read_BIDS_data(PATH_RUN, BIDS_EXAMPLE_PATH)

    settings_wrapper.test_settings()

    # read df_M1 / create M1 if None specified
    settings_wrapper.set_M1(m1_path=None, ch_names=raw_arr.ch_names,
                            ch_types=raw_arr.get_channel_types())
    settings_wrapper.set_fs_line_noise(fs, line_noise)

    # initialize generator for run function
    gen = generator.ieeg_raw_generator(raw_arr_data, settings_wrapper.settings)

    ieeg_batch = next(gen, None)

    return ieeg_batch, settings_wrapper.df_M1, settings_wrapper.settings, settings_wrapper.settings["fs"]


def initialize_rereference(df_M1):
    """The rereference class get's here instantiated given the supplied df_M1 table

    Args:
        df_M1 (pd Dataframe): rereference specifying table

    Returns:
        RT_rereference: Rereference object
    """
    ref_here = rereference.RT_rereference(df_M1, split_data=False)
    return ref_here


def test_rereference(ref_here, ieeg_batch, df_M1):
    """
    Args:
        ref_here (RT_rereference): Rereference initialized object
        ieeg_batch (np.ndarray): sample data
        df_M1 (pd.Dataframe): rereferencing dataframe
    """
    ref_dat = ref_here.rereference(ieeg_batch)

    print("Testing channels which are used but not rereferenced.")
    for no_ref_idx in np.where((df_M1.rereference == "None") & df_M1.used == 1)[0]:
        assert_array_equal(ref_dat[no_ref_idx, :], ieeg_batch[no_ref_idx, :])

    print("Testing ECOG average reference.")
    for ecog_ch_idx in np.where((df_M1['type'] == 'ecog') & (df_M1.rereference == 'average'))[0]:
        assert_array_equal(ref_dat[ecog_ch_idx, :], ieeg_batch[ecog_ch_idx, :] -
                           ieeg_batch[(df_M1['type'] == 'ecog') & (df_M1.index != ecog_ch_idx)].mean(axis=0))

    print("Testing bipolar reference.")
    for bp_reref_idx in [ch_idx for ch_idx, ch in
                         enumerate(df_M1.rereference) if ch in list(df_M1.name)]:
        # bp_reref_idx is the channel index of the rereference anode
        # referenced_bp_channel is the channel index which is the rereference cathode
        referenced_bp_channel = np.where(df_M1.iloc[bp_reref_idx]['rereference'] == df_M1.name)[0][0]
        assert_array_equal(ref_dat[bp_reref_idx, :],
                           ieeg_batch[bp_reref_idx, :] - ieeg_batch[referenced_bp_channel, :])


def test_sharpwaves(data, features_, ch, example_settings, fs):

    print("Initializing sharp wave test object.")
    sw_features = sharpwaves.SharpwaveAnalyzer(example_settings["sharpwave_analysis_settings"], fs)
    print("Estimating sharp wave features.")
    features_ = sw_features.get_sharpwave_features(features_, data, ch)

def test_BIDS_feature_estimation():
    print("Testing feature estimation for example data.")
    example_BIDS.run_example_BIDS()

def test_ML_features():
    print("Testing machine learning feature estimation.")
    example_ML.run_example_ML()

def test_feature_read_out():
    print("Testing feature read out.")
    example_read_features.run_example_read_features()


if __name__ == "__main__":

    ieeg_batch, df_M1, example_settings, fs = read_example_data(PATH_PYNEUROMODULATION)
    ref_here = initialize_rereference(df_M1)
    test_rereference(ref_here, ieeg_batch, df_M1)
    # test sharpwaves feature estimation for specifc channel
    features_ = dict()
    ch = df_M1.name.iloc[0]

    # estimate only the last / new data coming from the new sampling frequency
    # this speeds up estimation
    index_sharpwave = int(fs / example_settings["sampling_rate_features"])
    features_ = test_sharpwaves(ieeg_batch[0, -index_sharpwave:], features_, ch, example_settings, fs)

    # running example files
    test_BIDS_feature_estimation()
    test_ML_features()
    test_feature_read_out()

    print("All tests passed through.")
