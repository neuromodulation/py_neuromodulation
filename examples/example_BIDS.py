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

    # write BIDS example path in settings.json
    PATH_SETTINGS = os.path.abspath('examples\\nm_settings.json')
    with open(PATH_SETTINGS, encoding='utf-8') as json_file:
        settings = json.load(json_file)
    settings["BIDS_path"] = os.path.abspath('examples\\data')

    # write relative feature output folder
    settings["out_path"] = os.path.abspath(os.path.join('examples', 'data', 'derivatives'))
    with open('examples\\nm_settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

    PATH_RUN = os.path.join(
        os.path.abspath('examples\\data'), 'sub-testsub', 'ses-EphysMedOff', 'ieeg',
        "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")

    nm_start_BIDS.est_features_run(PATH_RUN, PATH_SETTINGS=PATH_SETTINGS)
