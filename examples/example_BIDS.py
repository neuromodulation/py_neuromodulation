import multiprocessing
import sys 
from bids import BIDSLayout
from itertools import product

sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
import start_BIDS

if __name__ == "__main__":

    PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    run_files = layout.get(extension='.vhdr')
    M1_files = [None for i in range(len(run_files))]  # specify no M1 files

    # example single run file estimation
    start_BIDS.est_features_run(run_files[0])

    # multiprocessing cohort analysis    
    # pool = multiprocessing.Pool()
    # pool.map(start_BIDS.est_features_run, product(M1_files, run_files))