import os
import sys

from time import time as time

import mne_bids

sys.path.insert(
    0, r'C:\Users\richa\GitHub\py_neuromodulation\pyneuromodulation')
import start_BIDS


def get_all_files(path, suffix, get_bids=False, prefix=None, bids_root=None,
                  verbose=False, extension=None):
    """Return all files in all (sub-)directories of path with given suffixes and prefixes (case-insensitive).

    Args:
        path (string)
        suffix (iterable): e.g. ["vhdr", "edf"] or ".json"
        get_bids (boolean): True if BIDS_Path type should be returned instead of string. Default: False
        bids_root (string/path): Path of BIDS root folder. Only required if get_bids=True.
        prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)
        verbose (boolean): verbose level
        extension (string): any keyword to search for in filename
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


root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Beijing_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26'
deriv_root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Beijing_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26\derivatives'

files = get_all_files(
    path=root,
    suffix='vhdr',
    get_bids=True,
    prefix='ButtonPress',
    bids_root=root,
    verbose=True,
    extension=None)

### No Normalization
path_settings = os.path.join(deriv_root, 'feat_add_HFA_no_norm', 'settings.json')
for file in files[:]:
    path_df = os.path.join(
        deriv_root, file.update(extension=None).basename+'_m1.tsv')
    start = time()
    start_BIDS.est_features_run(
        file, PATH_M1=path_df, PATH_SETTINGS=path_settings, verbose=True)
    print("Elapsed time: ", time()-start, " seconds")

### 10s Normalization, median-zscore
path_settings = os.path.join(deriv_root, 'feat_add_HFA_10s_norm', 'settings.json')
for file in files[:]:
    path_df = os.path.join(
        deriv_root, file.update(extension=None).basename+'_m1.tsv')
    start = time()
    start_BIDS.est_features_run(
        file, PATH_M1=path_df, PATH_SETTINGS=path_settings, verbose=True)
    print("Elapsed time: ", time()-start, " seconds")

### 30s Normalization, median-zscore
path_settings = os.path.join(deriv_root, 'feat_add_HFA_30s_norm', 'settings.json')
for file in files[:]:
    path_df = os.path.join(
        deriv_root, file.update(extension=None).basename+'_m1.tsv')
    start = time()
    start_BIDS.est_features_run(
        file, PATH_M1=path_df, PATH_SETTINGS=path_settings, verbose=True)
    print("Elapsed time: ", time()-start, " seconds")
