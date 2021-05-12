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
    PATH_PYNEUROMODULATION = os.pardir
    
    # BERLIN SUBJECTS
    BIDS_PATH = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Data\BIDS_Berlin_ECOG_LFP\rawdata"
    PATH_RUNS  = [
        #r"sub-005\ses-EphysMedOff01\ieeg\sub-005_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-005\ses-EphysMedOff01\ieeg\sub-005_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-005\ses-EphysMedOn01\ieeg\sub-005_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-005\ses-EphysMedOn01\ieeg\sub-005_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"
    ]

    PATH_RUNS_BERLIN = []
    for path in PATH_RUNS:
        PATH_RUNS_BERLIN.append(os.path.join(BIDS_PATH, path))

    # run them in a pool 
    pool = multiprocessing.Pool(processes=4)
    pool.map(start_BIDS.est_features_run, PATH_RUNS_BERLIN)
