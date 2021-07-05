import os
import sys
sys.path.insert(
    0, r'C:\Users\richa\GitHub\py_neuromodulation\working_directory'
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne_bids

import classification
import ieeg


def get_all_files(path, suffix, get_bids=False, prefix=None, bids_root=None,
                  verbose=False, extension=None):
    """Return all files in all (sub-)directories of path with given suffixes and prefixes (case-insensitive).

    Args:
        path (string)
        suffix (iterable): e.g. ["vhdr", "edf"] or ".json"
        get_bids (boolean): True if BIDS_Path type should be returned instead of string. Default: False
        bids_root (string/path): Path of BIDS root folder. Only required if get_bids=True.
        prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)

    Returns:
        filepaths (list of strings or list of BIDS_Path)
    """

    if isinstance(suffix, str):
        suffix = [suffix]
    if isinstance(prefix, str):
        prefix = [prefix]

    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for suff in suffix:
                if file.endswith(suff.lower()):
                    if not prefix:
                        filepaths.append(os.path.join(root, file))
                    else:
                        for pref in prefix:
                            if pref.lower() in file.lower():
                                filepaths.append(os.path.join(root, file))

    bids_paths = filepaths
    if get_bids:
        if not bids_root:
            print(
                "Warning: No root folder given. Please pass bids_root parameter to create a complete BIDS_Path object.")
        bids_paths = []
        for filepath in filepaths:
            entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.BIDSPath(subject=entities["subject"],
                                              session=entities["session"],
                                              task=entities["task"],
                                              run=entities["run"],
                                              acquisition=entities[
                                                  "acquisition"],
                                              suffix=entities["suffix"],
                                              extension=extension,
                                              root=bids_root)
            except ValueError as err:
                print(
                    f"ValueError while creating BIDS_Path object for file {filepath}: {err}")
            else:
                bids_paths.append(bids_path)

    if verbose:
        if not bids_paths:
            print("No corresponding files found.")
        else:
            print('Corresponding files found:')
            for idx, file in enumerate(bids_paths):
                print(idx, ':', os.path.basename(file))

    return bids_paths


root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\PROJECT_motor_onset_results\pipeline-motor_intention_pred_2021-07-02'

files = get_all_files(path=root, suffix='vhdr', get_bids=True, prefix=None,
                      bids_root=root, verbose=True, extension=None)


### File 0
ind = 0
file = files[ind]
raw = mne_bids.read_raw_bids(file, verbose=False)
print(*raw.ch_names, sep='\n')
data = raw.get_data(picks=[ch for ch in raw.ch_names if 'ANALOG' in ch]).squeeze()
if np.abs(data.min()) > np.abs(data.max()):
    data = data * -1.
plt.plot(data.squeeze())
data.shape
data = data[:-2000]
y_corrected, onoff, y = ieeg.baseline_correction(
    y=data, method='baseline_als', param=[1e2, 1e-4], thr=0.,
    normalize=True, Decimate=1, Verbose=True)
#plt.plot(y)
plt.plot(y_corrected)
plt.show()
y_corrected2, onoff, y = ieeg.baseline_correction(
    y=data.squeeze(), method='baseline_rope', param=1e1, thr=1e-1,
    normalize=True, Decimate=1, Verbose=True)
plt.plot(y_corrected2)
#plt.plot(y)
plt.show()