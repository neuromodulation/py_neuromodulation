import os
import sys
from time import time as time

import pandas as pd

import mne_bids

sys.path.insert(
    0, r'C:\Users\richa\GitHub\py_neuromodulation\pyneuromodulation')
import nm_reader as NM_reader
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


root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\PROJECT_motor_onset_results\pipeline-motor_intention_pred_2021-07-02'
deriv_root = os.path.join(root, 'derivatives', 'feat_EMG_RMS_200')

suffixes = [
    'add_HFA_no_norm_add_RMS',
    'add_HFA_10s_norm_add_RMS',
    'add_HFA_30s_norm_add_RMS']

for suffix in suffixes:
    root_berlin = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\PROJECT_motor_onset_results\pipeline-motor_intention_pred_2021-07-02\derivatives'
    feat_root = os.path.join(root_berlin, 'feat_' + suffix)
    feature_list = [
        "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-002_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg",
        "sub-003_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-003_ses-EphysMedOn03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOff02_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg",
        "sub-005_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg",
        "sub-005_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOn02_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg",
        "sub-005_ses-EphysMedOn02_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg",
        'sub-FOG008_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-FOG010_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-FOGC001_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg'
    ]
    for ft in feature_list[:]:
        rms_file = os.path.join(deriv_root, ft, ft + '_FEATURES.csv')
        df_rms = pd.read_csv(rms_file, index_col=0)
        feat_file = os.path.join(feat_root, ft, ft + '_FEATURES.csv')
        df_feat = pd.read_csv(feat_file, index_col=0)
        rms_cols = [col for col in df_rms.columns if 'RMS_200' in col]
        if rms_cols:
            print("Columns found:")
            print(*rms_cols, sep='\n')
            for col in rms_cols:
                df_feat[col] = df_rms[col]
            df_feat.to_csv(feat_file)
