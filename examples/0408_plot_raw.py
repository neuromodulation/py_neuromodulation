import multiprocessing
import sys
from bids import BIDSLayout
from itertools import product
import os
import json
from pathlib import Path
from matplotlib import pyplot as plt

# first parent to get example folder, second py_neuromodulation folder
PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
sys.path.append(os.path.join(Path(__file__).absolute().parent.parent,'examples'))

import start_BIDS
import IO

PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin_VoluntaryMovement\sub-005\ses-EphysMedOff02\ieeg\sub-005_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg.vhdr"
BIDS_PATH = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin_VoluntaryMovement"

if __name__ == "__main__":


    raw_arr, _, _, _ = IO.read_BIDS_data(PATH_RUN, BIDS_PATH)
    l_ch = [ch for ch in raw_arr.ch_names if 'ECOG' in ch]
    l_ch.append('rota_squared')
    fig = raw_arr.pick(l_ch).plot(scalings='auto', lowpass=200, highpass=2, block=True)

    #start_BIDS.est_features_run(PATH_RUN)
    """
    raw_arr, _, _, _ = IO.read_BIDS_data(PATH_RUN, BIDS_PATH)

    l_ch = [ch for ch in raw_arr.ch_names if 'ECOG' in ch]
    l_ch.append('TTL_1_clean')

    os.environ['ETS_TOOLKIT'] = 'qt4'
    os.environ['QT_API'] = 'pyqt5'
    #plt.figure()
    fig = raw_arr.pick(l_ch).plot(scalings='auto', lowpass=200, highpass=2, block=True)
    #fig.show()
    print("ho")
    """
