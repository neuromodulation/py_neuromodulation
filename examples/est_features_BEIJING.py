import multiprocessing
import sys 
from bids import BIDSLayout
from itertools import product
import os
from pathlib import Path

PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
sys.path.append(os.path.join(Path(__file__).absolute().parent.parent,'examples'))

import start_BIDS

if __name__ == "__main__":

    # example single run file estimation
    

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    subjects = layout.get_subjects()
    run_files = []
    for sub in subjects:
        if sub != "FOG013":
            try:
                run_files.append(layout.get(subject=sub, task='ButtonPress', extension='.vhdr')[0])
            except:
                pass
    
    #for run_file in run_files:
    #    start_BIDS.est_features_run(run_file)

    pool = multiprocessing.Pool(processes=20)
    
    # call here the pool only with run files, M1 files are created on the fly
    pool.map(start_BIDS.est_features_run, run_files)
