import multiprocessing
import sys
from bids import BIDSLayout
from itertools import product
import os
import json
from pathlib import Path

# first parent to get example folder, second py_neuromodulation folder
PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
sys.path.append(os.path.join(Path(__file__).absolute().parent.parent,'examples'))

import start_BIDS

def run_example_BIDS():
    """run the example BIDS path in pyneuromodulation/tests/data
    """

    print(os.getcwd())

    BIDS_EXAMPLE_PATH = os.path.abspath(
        os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation', 'tests',
                     'data'))

    # write BIDS example path in settings.json
    with open(os.path.join(PATH_PYNEUROMODULATION, 'examples',
                           'settings.json'), encoding='utf-8') as json_file:
        settings = json.load(json_file)
    settings["BIDS_path"] = BIDS_EXAMPLE_PATH

    # write relative feature output folder
    settings["out_path"] = os.path.abspath(
        os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation', 'tests',
                     'data', 'derivatives'))
    with open(os.path.join(
            PATH_PYNEUROMODULATION, 'examples', 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)

    PATH_RUN = os.path.join(
        BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff', 'ieeg',
        "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")

    start_BIDS.est_features_run(PATH_RUN)

if __name__ == "__main__":

    run_example_BIDS()