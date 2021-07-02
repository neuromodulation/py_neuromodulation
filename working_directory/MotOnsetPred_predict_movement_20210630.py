import os
import sys
sys.path.insert(
    0, r'C:\Users\richa\GitHub\py_neuromodulation\pyneuromodulation')

import numpy as np
import pandas as pd

import mne_bids

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


def run_classification(
        feat_list, nm_reader_, classifier, target_beg, target_en, out_root):
    """"""
    for feature_file in feat_list:
        print(feature_file)
        features_ = nm_reader_.read_features(feature_file)
        settings_ = nm_reader_.read_settings(feature_file)
        channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
        try:
            label_ = nm_reader_.read_label('rota_squared')
        except:
            label_ = nm_reader_.read_label('EMG_squared')
        dat_label_ = label_.values
        diff_ = np.zeros_like(dat_label_, dtype=int)
        diff_[1:] = np.diff(dat_label_)
        events_ = np.nonzero(diff_)[0]
        feat_picks = ['bandpass_activity_theta',
                      'bandpass_activity_alpha',
                      'bandpass_activity_low beta',
                      'bandpass_activity_high beta',
                      'bandpass_activity_low gamma',
                      'bandpass_activity_high gamma',
                      'bandpass_activity_high frequency activity']
        column_picks = [
            col for col in features_.columns if
            any([pick in col for pick in feat_picks])]
        features_ = features_[column_picks]
        feat_list = list()
        feat_list.append(features_)
        i = 1
        for s in np.arange(1, i):
            feat_list.append(features_.shift(s, axis=0))
        features_ = pd.concat(feat_list, axis=1)
        features_ = features_.fillna(0.)
        if target_en == "MovementEnd":
            prefix = '_movement_'
        else:
            prefix = '_mot_intention_'
        out_name = feature_file + prefix + classifier + '_results_' + \
                   str(i * 100) + 'ms.tsv'
        out_path = os.path.join(out_root, feature_file, out_name)
        classification.init_classification(
            features_, events_, channels_, target_beg, target_en, out_path,
            classifier=classifier, dist_onset=2., dist_end=2.)


def run_classification_diff(
        feat_list, nm_reader_, classifier, target_beg, target_en, optimize,
        balance, root, use_channels_):
    """"""
    for feature_file in feat_list:
        print("File: ", feature_file)
        features_ = nm_reader_.read_features(feature_file)
        settings_ = nm_reader_.read_settings(feature_file)
        channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
        try:
            label_ = nm_reader_.read_label('rota_squared')
        except:
            label_ = nm_reader_.read_label('EMG_squared')
        dat_label_ = label_.values
        diff_ = np.zeros_like(dat_label_, dtype=int)
        diff_[1:] = np.diff(dat_label_)
        events_ = np.nonzero(diff_)[0]
        feat_picks = ['bandpass_activity_theta',
                      'bandpass_activity_alpha',
                      'bandpass_activity_low beta',
                      'bandpass_activity_high beta',
                      'bandpass_activity_low gamma',
                      'bandpass_activity_high gamma',
                      'bandpass_activity_high frequency activity']
        column_picks = [
            col for col in features_.columns if
            any([pick in col for pick in feat_picks])]
        features_ = features_[column_picks]
        feat_list = list()
        feat_list.append(features_)
        feat_list.append(features_.diff(axis=0, periods=1).rename(
            columns={col: col + '_diff' for col in features_.columns}))
        i = 1
        for s in np.arange(1, i):
            feat_list.append(features_.shift(s, axis=0))
        features_ = pd.concat(feat_list, axis=1)
        features_ = features_.fillna(0.)
        #print('Features: ', features_.shape)
        if target_en == "MovementEnd":
            prefix = '_movement_'
        else:
            prefix = '_mot_intention_'
        if use_channels_ == 'all':
            prefix += 'all_channels_'
        opt_str = '_opt_' if optimize else '_no_opt_'
        out_name = feature_file + prefix + classifier + opt_str + 'results_' \
                   + str(i * 100) + 'ms' + '_diff' + '.tsv'
        out_path = os.path.join(root, feature_file, out_name)
        classification.init_classification(
            features_, events_, channels_, target_beg, target_en, out_path,
            classifier=classifier, dist_onset=2., dist_end=2.,
            optimize=optimize, balance=balance, use_channels=use_channels_)


def run_prediction(
        feat_list, nm_reader_, classifier, target_beg, target_en, optimize,
        balance, root, use_channels_):
    """"""
    for feature_file in feat_list:
        print("File: ", feature_file)
        features_ = nm_reader_.read_features(feature_file)
        settings_ = nm_reader_.read_settings(feature_file)
        channels_ = np.array(settings_['ch_names'])[settings_['feature_idx']]
        try:
            label_ = nm_reader_.read_label('rota_squared')
        except:
            label_ = nm_reader_.read_label('EMG_squared')
        targets = ['analog', 'rota', 'rms', 'emg_squared']
        #targets = ['rota_squared', 'emg_squared']
        i = 0
        target_ = pd.DataFrame()
        while len(target_.columns) == 0:
            target_pick = targets[i]
            col_picks = [
                col for col in features_.columns if target_pick in col.lower()]
            for col in col_picks:
                target_[col] = features_[col]
            i += 1
        assert len(col_picks) == 1, f"Multiple targets found: {col_picks}"
        dat_label_ = label_.values
        diff_ = np.zeros_like(dat_label_, dtype=int)
        diff_[1:] = np.diff(dat_label_)
        events_ = np.nonzero(diff_)[0]
        feat_picks = ['bandpass_activity_theta',
                      'bandpass_activity_alpha',
                      'bandpass_activity_low beta',
                      'bandpass_activity_high beta',
                      'bandpass_activity_low gamma',
                      'bandpass_activity_high gamma',
                      'bandpass_activity_high frequency activity']
        column_picks = [
            col for col in features_.columns if
            any([pick in col for pick in feat_picks])]
        features_ = features_[column_picks]
        feat_list = list()
        feat_list.append(features_)
        feat_list.append(features_.diff(axis=0, periods=1).rename(
            columns={col: col + '_diff' for col in features_.columns}))
        i = 1
        for s in np.arange(1, i):
            feat_list.append(features_.shift(s, axis=0))
        features_ = pd.concat(feat_list, axis=1)
        features_ = features_.fillna(0.)
        #print('Features: ', features_.shape)
        if target_en == "MovementEnd":
            prefix = '_movement_'
        else:
            prefix = '_mot_intention_'
        if use_channels_ == 'all':
            prefix += 'all_channels_'
        opt_str = '_opt_' if optimize else '_no_opt_'
        out_name = feature_file + prefix + classifier + opt_str + 'results_' \
                   + str(i * 100) + 'ms' + '_diff' + '.tsv'
        out_path = os.path.join(root, feature_file, out_name)
        classification.init_prediction(
            features_, target_, events_, channels_, target_beg, target_en,
            out_path, classifier=classifier, dist_onset=2., dist_end=2.,
            optimize=optimize, balance=balance, use_channels=use_channels_,
            pred_begin=-4., pred_end=3.)


suffixes = ['add_HFA_no_norm', 'add_HFA_10s_norm', 'add_HFA_30s_norm']
suffixes = ['add_HFA_10s_norm']
for suffix in suffixes:

    ### BERLIN
    root_berlin = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Berlin_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26'
    deriv_root_berlin = os.path.join(root_berlin, 'derivatives', 'feat_' + suffix)
    nm_reader_berlin = NM_reader.NM_Reader(deriv_root_berlin)
    #feature_list_berlin = nm_reader_berlin.get_feature_list()
    #print(*feature_list_berlin, sep='\n')
    feature_list_berlin =[
        "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-002_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        #"sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg",
        "sub-003_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        #"sub-003_ses-EphysMedOn03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        #"sub-004_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        #"sub-004_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        "sub-004_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        "sub-005_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        #"sub-005_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg",
        #"sub-005_ses-EphysMedOff02_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg",
        #"sub-005_ses-EphysMedOff02_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg",
        "sub-005_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg",
        #"sub-005_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"
        #"sub-005_ses-EphysMedOn02_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg",
        #"sub-005_ses-EphysMedOn02_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg"
    ]

    ### BEIJING
    root_beijing = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\Beijing_ECOG_LFP_derivatives\pipeline-MotOnsetPred_2021-04-26'
    deriv_root_beijing = os.path.join(root_beijing, 'derivatives', 'feat_' + suffix)
    nm_reader_beijing = NM_reader.NM_Reader(deriv_root_beijing)
    #feature_list = nm_reader.get_feature_list()
    #print(*feature_list, sep='\n')
    feature_list_beijing = [
        #'sub-FOG006_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-FOG008_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-FOG010_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg',
        #'sub-FOG013_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-FOGC001_ses-EphysMedOff_task-ButtonPress_acq-StimOff_run-01_ieeg'
    ]

    #classifiers = ['lda', 'lr', 'catboost', 'xgb', 'lin_svm', 'svm_lin', 'svm_rbf',
     #              'svm_poly', 'svm_sig']
    #classifiers = ['lda_oversample', 'lr_oversample', 'catboost_oversample',
     #              'xgb_oversample', 'lin_svm_oversample',
      #             'svm_lin_oversample', 'svm_rbf_oversample',
       #            'svm_poly_oversample', 'svm_sig_oversample']
    classifiers = ['lr_balance_weights', 'catboost_balance_weights',
                   'lin_svm_balance_weights', 'svm_lin_balance_weights',
                   'svm_rbf_balance_weights', 'svm_poly_balance_weights',
                   'xgb_balance_weights', 'svm_sig_balance_weights'
                   ]
    classifiers = ['lr_balance_weights']
    results_root = r'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\PROJECT_motor_onset_results'
    out_root = os.path.join(results_root, 'MotOnsetPred_2021-06-10_' + suffix)
    targets = [(0., "MovementEnd"), (-1., 0.)]
    feature_lists = [feature_list_berlin, feature_list_beijing]
    nm_readers = [nm_reader_berlin, nm_reader_beijing]

    for clf in classifiers:
        opt = True if 'opt' in clf else False
        print('Classifier: ', clf)
        if 'weight' in clf:
            balance = 'weight'
        elif 'undersamp' in clf:
            balance = 'undersample'
        else:
            balance = 'oversample'
        for target_begin, target_end in targets[:]:
            print('target_begin, target_end: ', target_begin, target_end)
            for feature_list, nm_reader in zip(feature_lists, nm_readers):
                run_prediction(
                    feature_list, nm_reader, clf, target_begin, target_end,
                    opt, balance, out_root, use_channels_='all')
