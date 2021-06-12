from colorama import Fore, Style
from operator import itemgetter
import os

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             log_loss)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC, SVC
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


def classify_catboost(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(max_depth, learning_rate, bagging_temperature, l2_leaf_reg,
                random_strength):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            groups_split = group_train[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.75, random_state=41)
            for train_ind, val_ind in val_inner_split.split(
                    X_tr, y_tr, groups_split):
                X_tr, X_va = X_tr[train_ind], X_tr[val_ind]
                y_tr, y_va = y_tr[train_ind], y_tr[val_ind]
                eval_set_inner = [(X_va, y_va)]
                if np.mean(y_tr) != 0.5:
                    X_tr, y_tr = balance_samples(
                        X_tr, y_tr, 'oversample')
            inner_model = CatBoostClassifier(
                iterations=100, loss_function='MultiClass', verbose=False,
                eval_metric="MultiClass", max_depth=round(max_depth),
                learning_rate=learning_rate,
                bagging_temperature=bagging_temperature,
                l2_leaf_reg=l2_leaf_reg, random_strength=random_strength)
            inner_model.fit(
                X_tr, y_tr, eval_set=eval_set_inner,
                early_stopping_rounds=25, verbose=False)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(bo_tune, {'max_depth': (4, 10),
                                            'learning_rate': (0.003, 0.3),
                                            'bagging_temperature': (0.0, 1.0),
                                            'l2_leaf_reg': (1, 30),
                                            'random_strength': (0.01, 1.)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        params = bo.max['params']
        params['max_depth'] = round(params['max_depth'])
        model = CatBoostClassifier(
            iterations=200, loss_function='MultiClass', verbose=False,
            use_best_model=True, eval_metric="MultiClass",
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_strength=params['random_strength'],
            bagging_temperature=params['bagging_temperature'],
            l2_leaf_reg=params['l2_leaf_reg'])
    else:
        # Use default values
        model = CatBoostClassifier(
            loss_function='MultiClass', verbose=False,
            use_best_model=True, eval_metric="MultiClass")
    # Train outer model
    val_split = GroupShuffleSplit(n_splits=1, train_size=0.75)
    for train_ind, val_ind in val_split.split(X_train, y_train, group_train):
        X_train, X_val = X_train[train_ind], X_train[val_ind]
        y_train, y_val = y_train[train_ind], y_train[
            val_ind]
        eval_set = [(X_val, y_val)]
        if np.mean(y_train) != 0.5:
            X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=25,
              verbose=False)
    y_pred = model.predict(X_test)

    return balanced_accuracy_score(y_test, y_pred)


def classify_lda(X_train, X_test, y_train, y_test):
    """"""
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(
            X_train, y_train, 'oversample')
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_lr(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = LogisticRegression(
                solver='newton-cg', C=C, max_iter=int(max_iter))
            inner_model.fit(X_tr, y_tr)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (0.1, 10.0), 'max_iter': (100, 1000)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = LogisticRegression(
            solver='newton-cg', max_iter=params['max_iter'], C=params['C'])
    else:
        # use default values
        model = LogisticRegression(solver='newton-cg')
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_lin_svm(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = LinearSVC(penalty='l2', fit_intercept=True,
                                    C=C, max_iter=max_iter, tol=tol,
                                    gamma='scale', shrinking=True,
                                    class_weight=None, probability=True,
                                    verbose=False)
            inner_model.fit(X_tr, y_tr)
            #cal = CalibratedClassifierCV(base_estimator=inner_model,
             #                            cv='prefit')
            #cal.fit(X_tr, y_tr)
            #y_probs = cal.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (1e-1, 1e1), 'max_iter': (100, 1000),
                      'tol': (1e-5, 1e-3)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = LinearSVC(penalty='l2', fit_intercept=True, C=params['C'],
                          max_iter=params['max_iter'], tol=params['tol'],
                          shrinking=True, class_weight=None,
                          verbose=False)
    else:
        # Use default values
        model = LinearSVC(penalty='l2', fit_intercept=True, class_weight=None,
                          verbose=False)
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_svm_lin(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = SVC(kernel='linear', C=C, max_iter=max_iter, tol=tol,
                              gamma='scale', shrinking=True, class_weight=None,
                              probability=True, verbose=False)
            inner_model.fit(X_tr, y_tr)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (pow(10, -1), pow(10, 1)), 'max_iter': (100, 1000),
                      'tol': (1e-4, 1e-2)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = SVC(kernel='linear', C=params['C'], max_iter=params['max_iter'],
                    tol=params['tol'], gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    else:
        # Use default values
        model = SVC(kernel='linear', gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_svm_rbf(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = SVC(kernel='rbf', C=C, max_iter=max_iter, tol=tol,
                              gamma='scale', shrinking=True, class_weight=None,
                              probability=True, verbose=False)
            inner_model.fit(X_tr, y_tr)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (pow(10, -1), pow(10, 1)), 'max_iter': (100, 1000),
                      'tol': (1e-4, 1e-2)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = SVC(kernel='rbf', C=params['C'], max_iter=params['max_iter'],
                    tol=params['tol'], gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    else:
        # Use default values
        model = SVC(kernel='rbf', gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_svm_poly(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = SVC(kernel='poly', C=C, max_iter=max_iter, tol=tol,
                              gamma='scale', shrinking=True, class_weight=None,
                              probability=True, verbose=False)
            inner_model.fit(X_tr, y_tr)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (pow(10, -1), pow(10, 1)), 'max_iter': (100, 1000),
                      'tol': (1e-4, 1e-2)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = SVC(kernel='poly', C=params['C'], max_iter=params['max_iter'],
                    tol=params['tol'], gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    else:
        # Use default values
        model = SVC(kernel='poly', gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_svm_sig(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(C, max_iter, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            if np.mean(y_tr) != 0.5:
                X_tr, y_tr = balance_samples(X_tr, y_tr, 'oversample')
            inner_model = SVC(kernel='sigmoid', C=C, max_iter=max_iter,
                              tol=tol, gamma='auto', shrinking=True,
                              class_weight=None, probability=True,
                              verbose=False)
            inner_model.fit(X_tr, y_tr)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {'C': (pow(10, -1), pow(10, 1)), 'max_iter': (100, 1000),
                      'tol': (1e-4, 1e-2)})
        bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = bo.max['params']
        params['max_iter'] = int(params['max_iter'])
        model = SVC(kernel='sigmoid', C=params['C'], max_iter=params['max_iter'],
                    tol=params['tol'], gamma='auto', shrinking=True,
                    class_weight=None, verbose=False)
    else:
        # Use default values
        model = SVC(kernel='sigmoid', gamma='scale', shrinking=True,
                    class_weight=None, verbose=False)
    # Train outer model
    if np.mean(y_train) != 0.5:
        X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def classify_xgb(X_train, X_test, y_train, y_test, group_train, optimize):
    """"""
    def bo_tune(max_depth, gamma, learning_rate, subsample, colsample_bytree):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(n_splits=3, train_size=0.66,
                                     random_state=42)
        scores = list()
        for train_index, test_index in cv_inner.split(
                X_train, y_train, group_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            groups_split = group_train[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.8, random_state=41)
            for train_ind, val_ind in val_inner_split.split(
                    X_tr, y_tr, groups_split):
                X_tr, X_va = X_tr[train_ind], X_tr[val_ind]
                y_tr, y_va = y_tr[train_ind], y_tr[val_ind]
                if np.mean(y_tr) != 0.5:
                    X_tr, y_tr = balance_samples(
                        X_tr, y_tr, 'oversample')
            eval_set_inner = [(X_va, y_va)]
            inner_model = XGBClassifier(
                objective='binary:logistic', use_label_encoder=False,
                eval_metric='logloss', n_estimators=200, gamma=gamma,
                learning_rate=learning_rate, max_depth=int(max_depth),
                colsample_bytree=colsample_bytree, subsample=subsample)
            inner_model.fit(
                X_tr, y_tr, eval_set=eval_set_inner, early_stopping_rounds=10,
                verbose=False)
            y_probs = inner_model.predict_proba(X_te)
            #print('y_probs.shape', y_probs.shape, 'y_test.shape', y_test.shape)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)
    if optimize:
        # Perform Bayesian Optimization
        xgb_bo = BayesianOptimization(
            bo_tune, {'max_depth': (4, 10), 'gamma': (0, 1),
                          'learning_rate': (0.001, 0.3),
                          'colsample_bytree': (0.1, 1),
                          'subsample': (0.8, 1)})
        xgb_bo.maximize(init_points=8, n_iter=12, acq='ei')
        # Train outer model with optimized parameters
        params = xgb_bo.max['params']
        params['max_depth'] = int(params['max_depth'])
        model = XGBClassifier(
            objective='binary:logistic', use_label_encoder=False,
            n_estimators=200, eval_metric='logloss', gamma=params['gamma'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'], subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'])
    else:
        # Use default values
        model = XGBClassifier(
            objective='binary:logistic', use_label_encoder=False,
            n_estimators=200, eval_metric='logloss')
    # Train outer model
    val_split = GroupShuffleSplit(n_splits=1, train_size=0.8)
    for train_ind, val_ind in val_split.split(X_train, y_train, group_train):
        X_train, X_val = X_train[train_ind], X_train[val_ind]
        y_train, y_val = y_train[train_ind], y_train[
            val_ind]
        eval_set = [(X_val, y_val)]
        if np.mean(y_train) != 0.5:
            X_train, y_train = balance_samples(X_train, y_train, 'oversample')
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10,
              verbose=False)
    y_pred = model.predict(X_test)

    return balanced_accuracy_score(y_test, y_pred)


def get_class_scores(features, labels, cross_val, classifier, groups):
    """"""
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
    """"""
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
        classifier='lda', dist_onset=2., dist_end=2., optimize=True):
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
    target_begin : int | float
        Begin of target to be used for classification. Use e.g. 0.0 for target
        begin at movement onset or e.g. -1.5 for classifying movement intention
        starting -1.5s before motor onset.
    target_end : int | float | 'MovementEnd'
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
    dist_end = 0. if any([exc in out_file for exc in exceptions]) \
            else dist_end
    data = features.values
    data, labels, events_used, groups = get_feat_array(
        data, events, sfreq=10,
        target_begin=target_begin, target_end=target_end,
        dist_onset=dist_onset, dist_end=dist_end)
    features = pd.DataFrame(data, columns=features.columns)
    cv_outer = GroupShuffleSplit(n_splits=5, train_size=0.8)
    results = list()
    fold = 0
    for train_ind, test_ind in cv_outer.split(features.values, labels, groups):
        print(Fore.LIGHTCYAN_EX + f"Fold no.: {fold}")
        print(Style.RESET_ALL)
        features_train, features_test = features.iloc[train_ind], \
                                        features.iloc[test_ind]
        y_train, y_test = labels[train_ind], labels[test_ind]
        groups_train = groups[train_ind]
        for ch_name in ch_names:
            print("Channel: ", ch_name)
            cols = [col for col in features_train.columns if ch_name in col]
            X_train = np.ascontiguousarray(features_train[cols].values)
            X_test = np.ascontiguousarray(features_test[cols].values)
            if 'catboost' in classifier:
                accuracy = classify_catboost(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'lda' in classifier:
                accuracy = classify_lda(
                    X_train, X_test, y_train, y_test)
            elif 'lin_svm' in classifier:
                accuracy = classify_lin_svm(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'lr' in classifier:
                accuracy = classify_lr(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'svm_lin' in classifier:
                accuracy = classify_svm_lin(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'svm_rbf' in classifier:
                accuracy = classify_svm_rbf(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'svm_poly' in classifier:
                accuracy = classify_svm_poly(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'svm_sig' in classifier:
                accuracy = classify_svm_sig(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            elif 'xgb' in classifier:
                accuracy = classify_xgb(
                    X_train, X_test, y_train, y_test, groups_train, optimize)
            else:
                raise ValueError(f"Classifier not found: {classifier}")
            results.append([accuracy, fold, ch_name])
        fold += 1
    curr_df = pd.DataFrame(data=results,
                           columns=['accuracy', 'fold', 'channel_name'])
    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
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
