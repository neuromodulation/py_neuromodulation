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

        self.nm_BIDS._set_run()
    
    def test_fooof_features(self):
        data = self.nm_BIDS.get_data()
        feature_series = self.nm_BIDS.run_analysis.process_data(data)
        assert feature_series is not None

def test_fooof():
    test_wrapper = TestWrapper()
    test_wrapper.test_fooof_features()
