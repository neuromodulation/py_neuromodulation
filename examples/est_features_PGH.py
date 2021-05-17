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

if __name__ == "__main__":

    with open(os.path.join(PATH_PYNEUROMODULATION, 'examples',
                           'settings.json'), encoding='utf-8') as json_file:
        settings = json.load(json_file)
    
    settings["BIDS_path"] = "C:\\Users\\ICN_admin\\OneDrive - Charité - Universitätsmedizin Berlin\\Data\\BIDS_Pittsburgh_Gripforce\\rawdata"
    
    # get all run paths for sub000
    layout = BIDSLayout(settings["BIDS_path"])
    run_files = layout.get(extension='.vhdr')
    for run_file in run_files:
        start_BIDS.est_features_run(run_file)

