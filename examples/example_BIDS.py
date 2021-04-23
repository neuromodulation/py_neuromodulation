import multiprocessing
import sys
from bids import BIDSLayout
from itertools import product
import os
import json

os.chdir(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir, 'examples'))
import start_BIDS

if __name__ == "__main__":

    print(os.getcwd())
    # example single run file estimation
    PATH_PYNEUROMODULATION = os.pardir

    BIDS_EXAMPLE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation',
                                     'tests', 'data')

    # write BIDS example path in settings.json
    with open(os.path.join(os.pardir, 'examples', 'settings.json'), encoding='utf-8') as json_file:
        settings = json.load(json_file)
    settings["BIDS_path"] = BIDS_EXAMPLE_PATH

    # write relative feature output folder
    settings["out_path"] = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation',
                                        'tests', 'data', 'derivatives')
    with open(os.path.join(os.pardir, 'examples', 'settings.json'), 'w') as f:
        json.dumps(settings, f, indent=4)

    PATH_RUN = os.path.join(BIDS_EXAMPLE_PATH, 'sub-testsub', 'ses-EphysMedOff',
                            'ieeg', "sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg.vhdr")

    start_BIDS.est_features_run(PATH_RUN)

    # multiprocessing cohort analysis
    '''
    PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Pittsburgh"
    layout = BIDSLayout(PATH_BIDS)
    #run_files = layout.get(extension='.vhdr')

    # get here only the first run file for every subject
    subjects = layout.get_subjects()
    run_files = []
    for sub in subjects:
        run_files.append(layout.get(subject=sub, extension='.vhdr')[0])

    #M1_files = [None for i in range(len(run_files))]  # specify no M1 files
    #start_BIDS.est_features_run(run_files[0])
    pool = multiprocessing.Pool(processes=50)

    # call here the pool only with run files, M1 files are created on the fly
    pool.map(start_BIDS.est_features_run, run_files)
    '''
