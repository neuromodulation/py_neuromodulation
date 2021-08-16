import multiprocessing
import sys
from bids import BIDSLayout
from itertools import product
import os
import json
from pathlib import Path
from pyneuromodulation import nm_start_BIDS


def run_example_BIDS():
    """run the example BIDS path in pyneuromodulation/tests/data
    """

    PATH_RUN = os.path.join(
        os.path.abspath('examples\\data'), 'sub-testsub', 'ses-EphysMedOff', 'ieeg',
        "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")

    # read default settings
    nm_BIDS = nm_start_BIDS.NM_BIDS(PATH_RUN)

    # add specific BIDS_PATH and out_path
    nm_BIDS.settings_wrapper.settings["BIDS_path"] = os.path.abspath('examples\\data')
    nm_BIDS.settings_wrapper.settings["out_path"] = os.path.abspath(os.path.join('examples', 'data', 'derivatives'))

    nm_BIDS.run_bids()

    # plot features for ECoG channels
    