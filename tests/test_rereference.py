import os
import sys
import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path

from py_neuromodulation import nm_BidsStream

# https://stackoverflow.com/a/10253916/5060208
# despite that pytest needs to be envoked by python: python -m pytest tests/

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

        RUN_NAME = "sub-testsub_ses-EphysMedOff_task-buttonpress_run-0_ieeg.vhdr"
        PATH_RUN = os.path.join(
            os.path.abspath(os.path.join('examples', 'data')),
            'sub-testsub',
            'ses-EphysMedOff',
            'ieeg',
            RUN_NAME
        )
        PATH_BIDS = os.path.abspath(os.path.join('examples', 'data'))
        PATH_OUT = os.path.abspath(os.path.join('examples', 'data', 'derivatives'))

        # read default settings
        self.nm_BIDS = nm_BidsStream.BidsStream(
            PATH_RUN=PATH_RUN,
            PATH_BIDS=PATH_BIDS,
            PATH_OUT=PATH_OUT,
            LIMIT_DATA=False
        )
        self.nm_channels = self.nm_BIDS.nm_channels
        self.settings = self.nm_BIDS.settings

        self.ieeg_batch = self.nm_BIDS.get_data()

    def test_rereference(self):
        """
        Args:
            ref_here (RT_rereference): Rereference initialized object
            ieeg_batch (np.ndarray): sample data
            df_M1 (pd.Dataframe): rereferencing dataframe
        """
        ref_dat = self.nm_BIDS.rereference.rereference(self.ieeg_batch)

        print("Testing channels which are used but not rereferenced.")
        for no_ref_idx in np.where(
            (
                self.nm_channels.rereference == "None") & self.nm_channels.used == 1
            )[0]:
            assert_array_equal(
                ref_dat[no_ref_idx, :],
                self.ieeg_batch[no_ref_idx, :]
            )

        print("Testing ECOG average reference.")
        for ecog_ch_idx in np.where(
            (self.nm_channels['type'] == 'ecog') &
                (self.nm_channels.rereference == 'average')
            )[0]:
            assert_array_equal(
                ref_dat[ecog_ch_idx, :],
                self.ieeg_batch[ecog_ch_idx, :] - 
                    self.ieeg_batch[(self.nm_channels['type'] == 'ecog') &
                        (self.nm_channels.index != ecog_ch_idx)].mean(axis=0)
            )

        print("Testing bipolar reference.")
        for bp_reref_idx in [ch_idx for ch_idx, ch in
                             enumerate(self.nm_channels.rereference)
                             if ch in list(self.nm_channels.name)]:
            # bp_reref_idx is the channel index of the rereference anode
            # referenced_bp_channel is the channel index which is the rereference cathode
            referenced_bp_channel = np.where(
                self.nm_channels.iloc[bp_reref_idx]['rereference'] ==
                    self.nm_channels.name
            )[0][0]
            assert_array_equal(ref_dat[bp_reref_idx, :],
                self.ieeg_batch[bp_reref_idx, :] - self.ieeg_batch[referenced_bp_channel, :]
            )


def test_rereference():
    test_wrapper = TestWrapper()
    test_wrapper.test_rereference()
