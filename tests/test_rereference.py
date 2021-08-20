import json
import os
import sys
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import mne_bids
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).parent.parent, 'pyneuromodulation'))
sys.path.append(os.path.join(Path(__file__).parent.parent, 'examples'))
# https://stackoverflow.com/a/10253916/5060208
# despite that pytest needs to be envoked by python: python -m pytest tests/

from pyneuromodulation import nm_define_nmchannels, nm_generator, nm_rereference, nm_settings, \
    nm_sharpwaves, nm_IO
from examples import example_BIDS


class TestWrapper:

    def __init__(self):
        """This test function sets a data batch and automatic initialized M1 datafram

        Args:
            PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

        Returns:
            ieeg_batch (np.ndarray): (channels, samples)
            df_M1 (pd Dataframe): auto intialized table for rereferencing
            settings_wrapper (settings.py): settings.json
            fs (float): example sampling frequency
        """

        # read and test settings first to obtain BIDS path
        self.settings_wrapper = nm_settings.SettingsWrapper(
            settings_path=os.path.join(os.path.dirname(nm_define_nmchannels .__file__), 'nm_settings.json'))

        #self.settings_wrapper.settings['BIDS_path'] = os.path.join(os.path.dirname(example_BIDS.__file___), 'data')
        self.settings_wrapper.settings['BIDS_path'] = os.path.join(Path(__file__).parent.parent, 'examples', 'data')
        self.settings_wrapper.settings['out_path'] = os.path.join(
            self.settings_wrapper.settings['BIDS_path'], 'derivatives')

        PATH_RUN = os.path.join(self.settings_wrapper.settings['BIDS_path'], 'sub-testsub', 'ses-EphysMedOff',
                                'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_run-0_ieeg.vhdr")
        # read BIDS data
        self.raw_arr, self.raw_arr_data, self.fs, self.line_noise = \
            nm_IO.read_BIDS_data(PATH_RUN, self.settings_wrapper.settings['BIDS_path'])

        self.settings_wrapper.test_settings()

        # read df_M1 / create M1 if None specified
        self.settings_wrapper.set_nm_channels(nm_channels_path=None, ch_names=self.raw_arr.ch_names,
                                              ch_types=self.raw_arr.get_channel_types())
        self.settings_wrapper.set_fs_line_noise(self.fs, self.line_noise)

        # initialize generator for run function
        self.gen = nm_generator.ieeg_raw_generator(self.raw_arr_data, self.settings_wrapper.settings)

        self.ieeg_batch = next(self.gen, None)

        self.initialize_rereference()

    def initialize_rereference(self):
        """The rereference class get's here instantiated given the supplied df_M1 table

        Args:
            df_M1 (pd Dataframe): rereference specifying table

        Returns:
            RT_rereference: Rereference object
        """
        self.ref_here = nm_rereference.RT_rereference(self.settings_wrapper.nm_channels, split_data=False)

    def test_rereference(self):
        """
        Args:
            ref_here (RT_rereference): Rereference initialized object
            ieeg_batch (np.ndarray): sample data
            df_M1 (pd.Dataframe): rereferencing dataframe
        """
        ref_dat = self.ref_here.rereference(self.ieeg_batch)

        print("Testing channels which are used but not rereferenced.")
        for no_ref_idx in np.where((self.settings_wrapper.nm_channels.rereference == "None") &
                                   self.settings_wrapper.nm_channels.used == 1)[0]:
            assert_array_equal(ref_dat[no_ref_idx, :], self.ieeg_batch[no_ref_idx, :])

        print("Testing ECOG average reference.")
        for ecog_ch_idx in np.where((self.settings_wrapper.nm_channels['type'] == 'ecog') &
                                    (self.settings_wrapper.nm_channels.rereference == 'average'))[0]:
            assert_array_equal(ref_dat[ecog_ch_idx, :], self.ieeg_batch[ecog_ch_idx, :] -
                               self.ieeg_batch[(self.settings_wrapper.nm_channels['type'] == 'ecog') &
                               (self.settings_wrapper.nm_channels.index != ecog_ch_idx)].mean(axis=0))

        print("Testing bipolar reference.")
        for bp_reref_idx in [ch_idx for ch_idx, ch in
                             enumerate(self.settings_wrapper.nm_channels.rereference)
                             if ch in list(self.settings_wrapper.nm_channels.name)]:
            # bp_reref_idx is the channel index of the rereference anode
            # referenced_bp_channel is the channel index which is the rereference cathode
            referenced_bp_channel = np.where(self.settings_wrapper.nm_channels.iloc[bp_reref_idx]['rereference'] ==
                                             self.settings_wrapper.nm_channels.name)[0][0]
            assert_array_equal(ref_dat[bp_reref_idx, :],
                               self.ieeg_batch[bp_reref_idx, :] - self.ieeg_batch[referenced_bp_channel, :])


def test_run():
    test_wrapper = TestWrapper()
    test_wrapper.test_rereference()