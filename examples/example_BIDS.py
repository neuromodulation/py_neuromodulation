import multiprocessing
import sys 
from bids import BIDSLayout
from itertools import product
import os

os.chdir(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir, 'examples'))
import start_BIDS

if __name__ == "__main__":

    # example single run file estimation
    PATH_PYNEUROMODULATION = os.pardir

    BIDS_EXAMPLE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation',
                                     'tests', 'data')

    PATH_RUN = os.path.join(BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff',
                            'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")

    start_BIDS.est_features_run(PATH_RUN)

    # multiprocessing cohort analysis
    '''
    PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    run_files = layout.get(extension='.vhdr')
    M1_files = [None for i in range(len(run_files))]  # specify no M1 files
    pool = multiprocessing.Pool()
    pool.map(start_BIDS.est_features_run, product(M1_files, run_files))
    '''
