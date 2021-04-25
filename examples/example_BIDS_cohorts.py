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

    # BIDS EXAMPLE FROM PY_NEUROMODULAITON
    '''
    BIDS_EXAMPLE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation',
                                     'tests', 'data')
    PATH_RUN = os.path.join(BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff',
                            'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")
    start_BIDS.est_features_run(PATH_RUN)
    '''
    
    # PITTSBURGH SUBJECTS
    '''
    # ALL RUNS PITTSBURGH
    PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Pittsburgh"
    layout = BIDSLayout(PATH_BIDS)
    run_files = layout.get(extension='.vhdr')
    '''

    '''
    # ONLY ONE RUN PER SUBJECT PITTSBURGH
    # get here only the first run file for every subject
    subjects = layout.get_subjects()
    run_files = []
    for sub in subjects:
        run_files.append(layout.get(subject=sub, extension='.vhdr')[0])
    '''

    # BEIJING SUBJECTS 
    PATH_BIDS =  "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    subjects = layout.get_subjects()
    run_files = []
    for sub in subjects:
        if sub != "FOG013":
            try:
                run_files.append(layout.get(subject=sub, task='ButtonPress', extension='.vhdr')[0])
            except:
                pass

    #M1_files = [None for i in range(len(run_files))]  # specify no M1 files
    #start_BIDS.est_features_run(run_files[4])
    pool = multiprocessing.Pool(processes=10)
    
    # call here the pool only with run files, M1 files are created on the fly
    pool.map(start_BIDS.est_features_run, run_files)
    
    
    # BERLIN SUBJECTS
    
    BIDS_PATH = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Data\BIDS_Berlin_ECOG_LFP\rawdata"
    PATH_RUNS  = [
        r"sub-001\ses-EphysMedOff01\ieeg\sub-001_ses-EphysMedOff01_task-BlockRotationR_acq-StimOffOn_run-01_ieeg.vhdr",
        r"sub-001\ses-EphysMedOn01\ieeg\sub-001_ses-EphysMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-002\ses-EphysMedOff02\ieeg\sub-002_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-003\ses-EphysMedOff01\ieeg\sub-003_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-004\ses-EphysMedOff01\ieeg\sub-004_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-004\ses-EphysMedOff01\ieeg\sub-004_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr",
        r"sub-005\ses-EphysMedOff01\ieeg\sub-005_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"
    ]

    #PATH_RUNS_BERLIN = []
    #for path in PATH_RUNS:
    #    PATH_RUNS_BERLIN.append(os.path.join(BIDS_PATH, path))
    
