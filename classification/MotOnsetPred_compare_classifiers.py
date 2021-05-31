import os
from importlib import reload
import sys
sys.path.insert(
    0, r'C:\Users\richa\GitHub\py_neuromodulation\pyneuromodulation')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import nm_reader as NM_reader
import classification


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

### BERLIN
deriv_root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Berlin_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26\derivatives\feat_no_norm'
nm_reader = NM_reader.NM_Reader(deriv_root)
feature_list = nm_reader.get_feature_list()
print(*feature_list, sep='\n')

### MOVEMENT ###
target_begin = 0.
target_end = "MovementEnd"
###################
for feature_file in feature_list[16:]:
    print(feature_file)
    features_ = nm_reader.read_features(feature_file)
    settings_ = nm_reader.read_settings(feature_file)
    channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
    label_ = nm_reader.read_label('rota_squared')
    dat_label_ = label_.values
    diff_ = np.zeros_like(dat_label_, dtype=int)
    diff_[1:] = np.diff(dat_label_)
    events_ = np.nonzero(diff_)[0]
    feat_picks = ['bandpass_activity_theta', 'bandpass_activity_alpha',
                  'bandpass_activity_low beta', 'bandpass_activity_high beta',
                  'bandpass_activity_low gamma', 'bandpass_activity_high gamma']
    column_picks = [
        col for col in features_.columns if any([pick in col for pick in feat_picks])]
    features_ = features_[column_picks]
    feat_list = list()
    feat_list.append(features_)
    i = 4
    for s in np.arange(1, i):
        feat_list.append(features_.shift(s, axis=0))
    features_ = pd.concat(feat_list, axis=1)
    features_ = features_.fillna(0.)
    out_name = feature_file + '_movement_XGB_results_' + str(i * 100) + 'ms.tsv'
    out_path = os.path.join(deriv_root, feature_file, out_name)
    classification.init_classification(
        features_, events_, channels_, target_begin, target_end, out_path, classifier='xgb')

### MOTOR ONSET ###
target_begin = -1.
target_end = 0.
###################
for feature_file in feature_list[0:]:
    print(feature_file)
    features_ = nm_reader.read_features(feature_file)
    settings_ = nm_reader.read_settings(feature_file)
    channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
    label_ = nm_reader.read_label('rota_squared')
    dat_label_ = label_.values
    diff_ = np.zeros_like(dat_label_, dtype=int)
    diff_[1:] = np.diff(dat_label_)
    events_ = np.nonzero(diff_)[0]
    feat_picks = ['bandpass_activity_theta', 'bandpass_activity_alpha',
                  'bandpass_activity_low beta', 'bandpass_activity_high beta',
                  'bandpass_activity_low gamma', 'bandpass_activity_high gamma']
    column_picks = [
        col for col in features_.columns if any([pick in col for pick in feat_picks])]
    features_ = features_[column_picks]
    feat_list = list()
    feat_list.append(features_)
    i = 4
    for s in np.arange(1, i):
        feat_list.append(features_.shift(s, axis=0))
    features_ = pd.concat(feat_list, axis=1)
    features_ = features_.fillna(0.)
    out_name = feature_file + '_mot_onset_XGB_results_' + str(i * 100) + 'ms.tsv'
    out_path = os.path.join(deriv_root, feature_file, out_name)
    classification.init_classification(
        features_, events_, channels_, target_begin, target_end, out_path, classifier='xgb')

### BEIJING
out_root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Beijing_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26'
deriv_root = os.path.join(out_root, 'derivatives', 'feat_no_norm')
nm_reader = NM_reader.NM_Reader(deriv_root)
#feature_list = nm_reader.get_feature_list()
#print(*feature_list, sep='\n')
feature_list = [
    'sub-FOG006_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
    'sub-FOG008_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
    'sub-FOG010_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg',
    #'sub-FOG013_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg',
    'sub-FOGC001_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg'
]
### MOVEMENT ###
target_begin = 0.
target_end = "MovementEnd"
###################
for feature_file in feature_list:
    print(feature_file)
    features_ = nm_reader.read_features(feature_file)
    settings_ = nm_reader.read_settings(feature_file)
    channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
    label_ = nm_reader.read_label('EMG_squared')
    dat_label_ = label_.values
    diff_ = np.zeros_like(dat_label_, dtype=int)
    diff_[1:] = np.diff(dat_label_)
    events_ = np.nonzero(diff_)[0]
    feat_picks = ['bandpass_activity_theta', 'bandpass_activity_alpha',
                  'bandpass_activity_low beta', 'bandpass_activity_high beta',
                  'bandpass_activity_low gamma', 'bandpass_activity_high gamma']
    column_picks = [
        col for col in features_.columns if any([pick in col for pick in feat_picks])]
    features_ = features_[column_picks]
    feat_list = list()
    feat_list.append(features_)
    i = 4
    for s in np.arange(1, i):
        feat_list.append(features_.shift(s, axis=0))
    features_ = pd.concat(feat_list, axis=1)
    features_ = features_.fillna(0.)
    out_name = feature_file + '_movement_XGB_results_' + str(i * 100) + 'ms.tsv'
    out_path = os.path.join(deriv_root, feature_file, out_name)
    classification.init_classification(
        features_, events_, channels_, target_begin, target_end, out_path, classifier='xgb')

### MOTOR ONSET ###
target_begin = -1.
target_end = 0.
###################
for feature_file in feature_list:
    print(feature_file)
    features_ = nm_reader.read_features(feature_file)
    settings_ = nm_reader.read_settings(feature_file)
    channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
    label_ = nm_reader.read_label('EMG_squared')
    dat_label_ = label_.values
    diff_ = np.zeros_like(dat_label_, dtype=int)
    diff_[1:] = np.diff(dat_label_)
    events_ = np.nonzero(diff_)[0]
    feat_picks = ['bandpass_activity_theta', 'bandpass_activity_alpha',
                  'bandpass_activity_low beta', 'bandpass_activity_high beta',
                  'bandpass_activity_low gamma', 'bandpass_activity_high gamma']
    column_picks = [
        col for col in features_.columns if any([pick in col for pick in feat_picks])]
    features_ = features_[column_picks]
    feat_list = list()
    feat_list.append(features_)
    i = 4
    for s in np.arange(1, i):
        feat_list.append(features_.shift(s, axis=0))
    features_ = pd.concat(feat_list, axis=1)
    features_ = features_.fillna(0.)
    out_name = feature_file + '_mot_onset_XGB_results_' + str(i * 100) + 'ms.tsv'
    out_path = os.path.join(deriv_root, feature_file, out_name)
    classification.init_classification(
        features_, events_, channels_, target_begin, target_end, out_path, classifier='xgb')