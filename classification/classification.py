from operator import itemgetter
import pprint

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             log_loss)
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier


def balance_samples(data, target, method='oversample'):
    """Balance class sizes to create equal class distributions.

    Parameters
    ----------
    data : array_like
        Feature array of shape (n_features, n_samples)
    target : array_like
        Array of class disribution of shape (n_samples, )
    method : {'oversample', 'undersample'}
        Method to be used for rebalancing classes. 'Oversample' will upsample
        the class with less samples. 'Undersample' will downsample the class
        with more samples. Default: 'oversample'

    Returns
    -------
    data : numpy.array
        Rebalanced feature array of shape (n_features, n_samples)
    target : numpy.array
        Corresponding class distributions. Class sizes are now evenly balanced.
    """
    if method == 'oversample':
        ros = RandomOverSampler(sampling_strategy='auto')
    elif method == 'undersample':
        ros = RandomUnderSampler(sampling_strategy='auto')  
    else:
        raise ValueError(f"Method not identified. Given method was {method}.")
    data, target = ros.fit_resample(data, target)
    return data, target


def classify_lda(features, events, ch_name, target_begin, target_end,
                 dist_onset, dist_end, verbose):
    """

    Parameters
    ----------
    features
    events
    ch_name
    target_begin
    target_end
    dist_onset
    dist_end
    verbose

    Returns
    -------

    """
    cols = [col for col in features.columns if ch_name in col]
    feat_picks = features[cols]
    data = feat_picks.values
    X, y, events_used, groups = get_feat_array(
        data, events, sfreq=10, 
        target_begin=target_begin, target_end=target_end,
        dist_onset=dist_onset, dist_end=dist_end)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    cv = GroupShuffleSplit(n_splits=5, train_size=0.8)
    accuracy, AP = get_class_scores(X, y, cv, clf, groups)
    if verbose:
        print(ch_name, ':', 'CV-AP: ', AP, 'CV-Accuracy: ', accuracy)
    return AP, accuracy


def classify_xgb(features, events, ch_name, target_begin, target_end,
                 dist_onset, dist_end, verbose):
    """

    Parameters
    ----------
    features
    events
    ch_name
    target_begin
    target_end
    dist_onset
    dist_end
    verbose

    Returns
    -------

    """
    def bo_tune_xgb(max_depth, gamma, learning_rate, subsample,
                    colsample_bytree):
        # Cross validating with the specified parameters in 5 folds
        scores = list()
        for train_index, test_index in cv_inner.split(
                features_train, labels_train, groups_inner):
            model = XGBClassifier(
                objective='multi:softprob', use_label_encoder=False,
                num_class=2, eval_metric='mlogloss', gamma=gamma,
                learning_rate=learning_rate, max_depth=int(max_depth),
                n_estimators=100, colsample_bytree=colsample_bytree,
                subsample=subsample)
            X_train, X_test = \
                features_train[train_index], features_train[test_index]
            y_train, y_test = labels_train[train_index], \
                              labels_train[test_index]
            groups_split = groups_inner[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.75, random_state=41)
            for train_ind, val_ind in val_inner_split.split(X_train, y_train,
                                                            groups_split):
                X_train, X_val = X_train[train_ind], X_train[val_ind]
                y_train, y_val = y_train[train_ind], y_train[val_ind]
                if np.mean(y_train) != 0.5:
                    X_train, y_train = balance_samples(
                        X_train, y_train, 'oversample')
            eval_set = [(X_val, y_val)]
            model.fit(X_train, y_train, eval_metric="mlogloss",
                      eval_set=eval_set,
                      early_stopping_rounds=25, verbose=False)
            y_probs = model.predict_proba(
                X_test, iteration_range=(0, model.best_iteration))
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_test, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    cols = [col for col in features.columns if ch_name in col]
    feat_picks = features[cols]
    data = feat_picks.values
    features, labels, events_used, groups = get_feat_array(
        data, events, sfreq=10,
        target_begin=target_begin, target_end=target_end,
        dist_onset=dist_onset, dist_end=dist_end)

    cv_outer = GroupShuffleSplit(n_splits=5, train_size=0.8)
    accuracy, AP = [], []
    # Perform Bayesian optimization once for each outer fold
    for train_ind, test_ind in cv_outer.split(features, labels, groups):
        features_train, features_test = features[train_ind], \
                                        features[test_ind]
        labels_train, labels_test = labels[train_ind], labels[test_ind]
        groups_train = groups[train_ind]
        groups_inner = groups[train_ind]
        cv_inner = GroupShuffleSplit(n_splits=5, train_size=0.8,
                                     random_state=42)
        # Now perform Bayesian Optimization
        xgb_bo = BayesianOptimization(
            bo_tune_xgb, {'max_depth': (4, 10), 'gamma': (0, 1),
                          'learning_rate': (0.001, 0.3),
                          'colsample_bytree': (0.1, 1),
                          'subsample': (0.8, 1)})
        xgb_bo.maximize(init_points=8, n_iter=7, acq='ei')
        # Now train outer model with optimized parameters
        params = xgb_bo.max['params']
        params['max_depth'] = int(params['max_depth'])
        model = XGBClassifier(objective='multi:softprob',
                              use_label_encoder=False,
                              num_class=2, eval_metric='mlogloss',
                              gamma=params['gamma'],
                              learning_rate=params['learning_rate'],
                              max_depth=params['max_depth'],
                              n_estimators=200,
                              colsample_bytree=params['colsample_bytree'],
                              subsample=params['subsample'])
        val_split = GroupShuffleSplit(n_splits=1, train_size=0.75)
        for train_ind, val_ind in val_split.split(
                features_train, labels_train, groups_train):
            features_train, features_val = features_train[train_ind], \
                                           features_train[val_ind]
            labels_train, labels_val = labels_train[train_ind], labels_train[
                val_ind]
            if np.mean(labels_train) != 0.5:
                features_train, labels_train = balance_samples(
                    features_train, labels_train, 'oversample')
        eval_set = [(features_val, labels_val)]
        model.fit(features_train, labels_train,
                  eval_metric=["merror", "mlogloss"],
                  eval_set=eval_set, early_stopping_rounds=25, verbose=False)
        labels_pred = model.predict(
            features_test, iteration_range=(0, model.best_iteration))
        acc = balanced_accuracy_score(labels_test, labels_pred)
        #labels_score = model.decision_function(features_test)
        #ap = average_precision_score(labels_test, labels_score)
        accuracy.append(acc)
        AP.append(0.)
    if verbose:
        print(ch_name, ':', 'CV-AP: ', np.mean(AP), 'CV-Accuracy: ',
              np.mean(accuracy))
    return np.mean(AP), np.mean(accuracy)


def classify_lr(features, events, ch_name, target_begin, target_end,
                dist_onset, dist_end, verbose):
    """

    Parameters
    ----------
    features
    events
    ch_name
    target_begin
    target_end
    dist_onset
    dist_end
    verbose

    Returns
    -------

    """
    def bo_tune(solver, C, max_iter):
        # Cross validating with the specified parameters in 5 folds
        scores = list()
        for train_index, test_index in cv_inner.split(
                features_train, labels_train, groups_inner):
            model = LogisticRegression(solver=solver, C=C, max_iter=max_iter)
            X_train, X_test = \
                features_train[train_index], features_train[test_index]
            y_train, y_test = labels_train[train_index], \
                              labels_train[test_index]
            if np.mean(y_train) != 0.5:
                X_train, y_train = balance_samples(
                    X_train, y_train, 'oversample')
            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_test, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    print("Channel:", ch_name)
    cols = [col for col in features.columns if ch_name in col]
    feat_picks = features[cols]
    data = feat_picks.values
    features, labels, events_used, groups = get_feat_array(
        data, events, sfreq=10,
        target_begin=target_begin, target_end=target_end,
        dist_onset=dist_onset, dist_end=dist_end)

    cv_outer = GroupShuffleSplit(n_splits=5, train_size=0.8)
    accuracy, AP = [], []
    # Perform Bayesian optimization once for each outer fold
    for train_ind, test_ind in cv_outer.split(features, labels, groups):
        features_train, features_test = features[train_ind], \
                                        features[test_ind]
        labels_train, labels_test = labels[train_ind], labels[test_ind]
        groups_train = groups[train_ind]
        groups_inner = groups[train_ind]
        if np.mean(labels_train) != 0.5:

            features_train = np.concatenate((
                features_train, np.expand_dims(groups_train, axis=1)), axis=1)
            features_train, labels_train = balance_samples(
                features_train, labels_train, 'oversample')
            groups_train = features_train[:, -1]
            features_train = features_train[:, :-1]
        cv_inner = GroupShuffleSplit(n_splits=5, train_size=0.8,
                                     random_state=42)
        # Now perform Bayesian Optimization
        estimator = LogisticRegression()
        spaces = {'solver': Categorical([
                          'newton-cg', 'saga']),
                  'C': Real(pow(10, -3), pow(10, 0), prior="log-uniform"),
                  'max_iter': Integer(100, 1000)}
        model = BayesSearchCV(estimator=estimator, search_spaces=spaces,
                              n_iter=15, scoring='neg_log_loss', refit=True,
                              cv=cv_inner, iid=True, verbose=0)
        model.fit(features_train, labels_train, groups_train)
        print("Best_params_: %s" % model.best_params_)
        # Now train outer model with optimized parameters
        labels_pred = model.predict(features_test)
        acc = balanced_accuracy_score(labels_test, labels_pred)
        labels_score = model.decision_function(features_test)
        ap = average_precision_score(labels_test, labels_score)
        accuracy.append(acc)
        AP.append(ap)
    if verbose:
        print(ch_name, ':', 'CV-AP: ', np.mean(AP), 'CV-Accuracy: ',
              np.mean(accuracy))
    return np.mean(AP), np.mean(accuracy)


def get_class_scores(features, labels, cross_val, classifier, groups):
    """

    Parameters
    ----------
    features
    labels
    cross_val
    classifier
    groups

    Returns
    -------

    """
    accuracy, average_precision = [], []
    for train, test in cross_val.split(features, labels, groups):
        X_train = features[train]
        y_train = labels[train]
        if np.mean(y_train) != 0.5:
            X_train, y_train = balance_samples(X_train, y_train, 'oversample')
        X_test = features[test]
        y_test = labels[test]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_score = classifier.decision_function(X_test)
        accuracy.append(balanced_accuracy_score(y_test, y_pred))
        average_precision.append(average_precision_score(y_test, y_score))
    return np.mean(accuracy), np.mean(average_precision)


def get_feat_array(
        data, events, sfreq, target_begin, target_end, dist_onset, dist_end):
    """

    Parameters
    ----------
    data
    events
    sfreq
    target_begin
    target_end
    dist_onset
    dist_end

    Returns
    -------

    """
    dist_onset = int(dist_onset * sfreq)
    dist_end = int(dist_end * sfreq)
    
    rest_begin, rest_end = int(-3. * sfreq), int(-2. * sfreq)
    target_begin = int(target_begin * sfreq)
    if target_end != 'MovementEnd':
        target_end = int(target_end * sfreq)

    X, y, events_used, group_list = [], [], [], []

    for i, ind in enumerate(np.arange(0, len(events), 2)):
        append = True
        if i == 0:
            if target_end == 'MovementEnd':
                data_rest = data[
                            events[ind] + rest_begin:events[ind] + rest_end]
                data_target = data[events[ind] + target_begin:events[ind + 1]]
            else:
                data_rest = \
                    data[events[ind] + rest_begin:events[ind] + rest_end]
                data_target = \
                    data[events[ind] + target_begin:events[ind] + target_end]
        elif (events[ind] - dist_onset) - (events[ind - 1] + dist_end) <= 0:
            append = False
        else:
            dist = (events[ind] - dist_onset) - (events[ind - 1] + dist_end)
            if dist >= 3.:
                rest_begin = int(-5. * sfreq)
            else:
                rest_begin = rest_end - dist
            if target_end == 'MovementEnd':
                data_rest = \
                    data[events[ind] + rest_begin:events[ind] + rest_end]
                data_target = \
                    data[events[ind] + target_begin:events[ind + 1]]
            else:
                data_rest = \
                    data[events[ind] + rest_begin:events[ind] + rest_end]
                data_target = \
                    data[
                    events[ind] + target_begin:events[ind] + target_end]
        if append:
            X.extend((data_rest, data_target))
            y.extend((np.zeros(len(data_rest)), np.ones(len(data_target))))
            events_used.append(ind)
            group_list.append(np.full((len(data_rest) + len(data_target)), i))
    return np.concatenate(X, axis=0).squeeze(), np.concatenate(y), \
        np.array(events_used), np.concatenate(group_list)


def init_classification(
        features, events, ch_names, target_begin, target_end, out_file,
        classifier='lda', dist_onset=2., dist_end=2.):
    """Calculate classification performance and write to *.tsv file.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of features to be used for classification, where each column
        is a different feature and each index a new sample.
    events : np.array
        Array of events of shape (n_trials * 2, ) where each even index
        [0, 2, 4, ...] marks the onset and each odd index [1, 3, 5, ...] marks
        the end of a trial (e.g. a movement).
    ch_names : list
        List of all channel names.
    movement_begin : int | float
        Begin of target to be used for classification. Use e.g. 0.0 for target
        begin at movement onset or e.g. -1.5 for classifying movement intention
        starting -1.5s before motor onset.
    movement_end : int | float | 'MovementEnd'
        End of target to be used for classification. Use 'MovementEnd' for
        target end at movement end or e.g. 0.0 for classifying movement
        intention up to motor onset.
    out_file : path | string
        Name and path of the file where classification performance is saved.
    classifier : {'lda', 'xgb', 'lr'}
        Method for classification. Use 'lda' for regularized shrinkage Linear
        Discriminant Analysis. Use 'xgb' for XGBoost classifier with
        hyperparameter optimization using Bayesian Optimization. Default is
        'lda'.
    dist_onset : int | float | default: 2.0
        Minimum distance before onset of current trial for label `rest`.
        Choosing a different value than 2.0 is currently not recommended for
        dist_onset.
    dist_end : int | float | default: 2.0
        Minimum distance after previous trial for label `rest`.

    Returns
    -------
    None

    Examples
    --------
    Following are two examples on how to employ init_classification. First
    example demonstrates how to classify movement itself using sLDA, whereas
    the second example demonstrates how motor intention could be classified
    using XGBoost.

    >>> init_classification(features=features_, events=events_,
    >>>     channels=channels_, target_begin=0., target_end="Movement_End",
    >>>     out_path='movement_classif_results.tsv', classifier='lda',
    >>>     dist_onset=2., dist_end=2.)

    >>> init_classification(features=features_, events=events_,
    >>>     channels=channels_, target_begin=0., target_end="Movement_End",
    >>>     out_path='motor_intention_classif_results.tsv', classifier='xgb',
    >>>     dist_onset=2., dist_end=3.)
    """
    exceptions = [
        'sub-FOG006_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg',
        'sub-003_ses-EphysMedOn03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']
    scores_list = list()
    dist_end = 0. if any([exc in out_file for exc in exceptions]) \
            else dist_end
    for ch_name in ch_names:
        if classifier == 'lda':
            AP, accuracy = classify_lda(
                features, events, ch_name, target_begin, target_end,
                dist_onset, dist_end, verbose=False)
        elif classifier == 'xgb':
            AP, accuracy = classify_xgb(
                features, events, ch_name, target_begin, target_end,
                dist_onset, dist_end, verbose=True)
        elif classifier == 'lr':
            AP, accuracy = classify_lr(
                features, events, ch_name, target_begin, target_end,
                dist_onset, dist_end, verbose=True)
        scores_list.append([AP, accuracy])
    curr_df = pd.DataFrame(data=scores_list, index=ch_names, 
                           columns=['average_precision', 'accuracy'])
    curr_df.to_csv(out_file, sep='\t')
    

def max_val(list_, ind):
    """Return maximum value and index from list of lists.

    Parameters
    ----------
    list_ : list
        List of lists.
    ind : int
        Index of each single list which should be compared for maximum value.

    Returns
    -------
    Tuple
        Index of list that contains maximum value and maximum value itself.
    """
    return max(enumerate(map(itemgetter(ind), list_)), key=itemgetter(1))
