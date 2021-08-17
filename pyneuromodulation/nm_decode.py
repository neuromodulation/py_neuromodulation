from sklearn import model_selection, metrics, linear_model
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.utils import class_weight

import pandas as pd
import os
import json
import numpy as np
import xgboost as xgb
import _pickle as cPickle


class Decoder:

    def __init__(self, feature_path, feature_file,
                 model=linear_model.LinearRegression(),
                 eval_method=metrics.r2_score,
                 cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                 threshold_score=True) -> None:
        """Imitialize here a feature file for processing
        Read settings.json nm_channels.csv and features.csv
        Read target label

        Parameters
        ----------
        feature_path : string
            path to feature output folders
        feature_file : string
            specific feature folder
        model : machine learning model
            model that utilizes fit and predict functions
        eval_method : sklearn metrics
            evaluation scoring method
        cv_method : sklearm model_selection method
        threshold_score : boolean
            if True set lower threshold at zero (useful for r2)
        """

        # for the bayesian opt. function the objective fct only uses a single parameter
        # coming from gp_miminimize
        # therefore declare other parameters to global s.t. the function ca acess those

        self.feature_path = feature_path
        self.feature_file = feature_file
        self.features = pd.read_csv(os.path.join(feature_path, feature_file,
                                    feature_file + "_FEATURES.csv"), header=0)

        self.nm_channels = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                       feature_file + "_nm_channels.csv"), header=0)
        self.used_chs = list(self.nm_channels[(self.nm_channels["target"] == 0) &
                             (self.nm_channels["used"] == 1)]["new_name"])

        with open(os.path.join(self.feature_path, feature_file,
                               feature_file + "_SETTINGS.json")) as f:
            self.settings = json.load(f)
        self.run_analysis = None

        # for simplicity, choose here only first one
        # lateron check laterality
        # try to set contralateral, if not take the first
        target_channels = list(self.nm_channels[self.nm_channels["target"] == 1]["name"])
        if len(target_channels) == 1:
            self.target_ch = target_channels[0]
        elif self.settings["sess_right"] is True:
            # check if contralateral left (optimal clean) channel exists
            left_targets = [t_ch for t_ch in target_channels if "LEFT" in t_ch]
            if len(left_targets) == 1:
                self.target_ch = left_targets[0]
            else:
                CLEAN_LEFT = [t_ch for t_ch in left_targets if ("CLEAN" in t_ch) or
                              ("squared" in t_ch)]
                if len(CLEAN_LEFT) == 1:
                    self.target_ch = CLEAN_LEFT[0]
                else:
                    # search for "clean" or "squared" channel without laterality included
                    ch_without_lat = [t_ch for t_ch in list(self.nm_channels[self.nm_channels["target"] == 1]["name"])
                                      if ("CLEAN" in t_ch) or ("squared" in t_ch)]
                    if len(ch_without_lat) != 0:
                        self.target_ch = ch_without_lat[0]
                    else:
                        # take first target
                        self.target_ch = self.nm_channels[self.nm_channels["target"] == 1]["name"].iloc[0]
        else:
            # left session
            # check if contralateral right (optimal clean) channel exists
            right_targets = [t_ch for t_ch in target_channels if "RIGHT" in t_ch]
            if len(right_targets) == 1:
                self.target_ch = right_targets[0]
            else:
                CLEAN_RIGHT = [t_ch for t_ch in right_targets if ("CLEAN" in t_ch) or
                               ("squared" in t_ch)]
                if len(CLEAN_RIGHT) == 1:
                    self.target_ch = CLEAN_RIGHT[0]
                else:
                    # search for "clean" or "squared" channel without laterality included
                    ch_without_lat = [t_ch for t_ch in list(self.nm_channels[self.nm_channels["target"] == 1]["name"])
                                      if ("CLEAN" in t_ch) or ("squared" in t_ch)]
                    if len(ch_without_lat) != 0:
                        self.target_ch = ch_without_lat[0]
                    else:
                        # take first target
                        self.target_ch = self.nm_channels[self.nm_channels["target"] == 1]["name"].iloc[0]

        # for classification dependin on the label, set to binary label
        self.label = np.nan_to_num(np.array(self.features[self.target_ch])) > 0.3
        self.data = np.nan_to_num(np.array(self.features[[col for col in self.features.columns
                                  if not (('time' in col) or (self.target_ch in col))]]))

        self.model = model
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.threshold_score = threshold_score

    def set_data_ind_channels(self):
        """specified channel individual data
        """
        self.ch_ind_data = {}
        for ch in self.used_chs:
            self.ch_ind_data[ch] = np.nan_to_num(np.array(self.features[[col for col in self.features.columns
                                                          if col.startswith(ch)]]))

    def run_CV_ind_channels(self, XGB=True):
        """run the CV for every specified channel

        Parameters
        ----------
        XGB (boolean):
            if true split data into additinal validation, and run class weighted CV
        """
        self.ch_ind_pr = {}
        for ch in self.used_chs:
            self.run_CV(self.ch_ind_data[ch], self.label, XGB)
            self.ch_ind_pr[ch] = {}
            self.ch_ind_pr[ch]["score_train"] = self.score_train
            self.ch_ind_pr[ch]["score_test"] = self.score_test
            self.ch_ind_pr[ch]["y_test"] = self.y_test
            self.ch_ind_pr[ch]["y_train"] = self.y_train
            self.ch_ind_pr[ch]["y_test_pr"] = self.y_test_pr
            self.ch_ind_pr[ch]["y_train_pr"] = self.y_train_pr
            self.ch_ind_pr[ch]["X_train"] = self.X_train
            self.ch_ind_pr[ch]["X_test"] = self.X_test

    def set_data_grid_points(self):
        """Read the run_analysis
        Projected data has the shape (samples, grid points, features)
        """

        PATH_ML_ = os.path.join(self.feature_path, self.feature_file, self.feature_file + "_run_analysis.p")
        with open(PATH_ML_, 'rb') as input:  # Overwrites any existing file.
            self.run_analysis = cPickle.load(input)

        # get active grid points
        self.active_gridpoints = np.where(np.sum(self.run_analysis.projection.proj_matrix_cortex, axis=1) != 0)[0]

        # set grid point feature names
        ch = self.run_analysis.features.ch_names[0]
        l_features = list(self.run_analysis.feature_arr.columns)
        self.feature_names = [f[len(ch)+1:] for f in l_features if f.startswith(ch)]

        # write data for every active grid point and run the cross validation
        self.grid_point_ind_data = {}
        for grid_point in self.active_gridpoints:
            # samples, features
            self.grid_point_ind_data[grid_point] = np.nan_to_num(self.run_analysis.proj_cortex_array[:, grid_point, :])

    def run_CV_grid_points(self, XGB=True):
        """run cross validation across grid points

        Parameters
        ----------
        XGB (boolean):
            if true split data into additinal validation, and run class weighted CV
        """
        self.gridpoint_ind_pr = {}
        for grid_point in self.active_gridpoints:
            self.run_CV(self.grid_point_ind_data[grid_point], self.label, XGB)
            self.gridpoint_ind_pr[grid_point] = {}
            self.gridpoint_ind_pr[grid_point]["score_train"] = self.score_train
            self.gridpoint_ind_pr[grid_point]["score_test"] = self.score_test
            self.gridpoint_ind_pr[grid_point]["y_test"] = self.y_test
            self.gridpoint_ind_pr[grid_point]["y_train"] = self.y_train
            self.gridpoint_ind_pr[grid_point]["y_test_pr"] = self.y_test_pr
            self.gridpoint_ind_pr[grid_point]["y_train_pr"] = self.y_train_pr
            self.gridpoint_ind_pr[grid_point]["X_train"] = self.X_train
            self.gridpoint_ind_pr[grid_point]["X_test"] = self.X_test

    def run_CV(self, data=None, label=None, XGB=True):
        """Evaluate model performance on the specified cross validation.
        If no data and label is specified, use whole feature class attributes.

        Parameters
        ----------
        data (np.ndarray):
            data to train and test with shape samples, features
        label (np.ndarray):
            label to train and test with shape samples, features
        XGB (boolean):
            if true split data into additinal validation, and run class weighted CV
        Returns
        -------
        cv_res : float
            mean cross validation result
        """

        if data is None:
            print("use all channel data as features")
            data = self.data
            label = self.label

        # if xgboost being used, might be necessary to set the params individually

        self.score_train = []
        self.score_test = []
        self.y_test = []
        self.y_train = []
        self.y_test_pr = []
        self.y_train_pr = []
        self.X_test = []
        self.X_train = []

        for train_index, test_index in self.cv_method.split(self.data):

            model_train = clone(self.model)
            X_train, y_train = data[train_index, :], label[train_index]
            X_test, y_test = data[test_index], label[test_index]

            if y_train.sum() == 0:  # only one class present
                continue

            # optionally split training data also into train and validation
            # for XGBOOST
            if XGB:
                X_train, X_val, y_train, y_val = \
                    model_selection.train_test_split(
                        X_train, y_train, train_size=0.8, shuffle=False)

                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced', y=y_train)

                model_train.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10, sample_weight=classes_weights,
                    verbose=False)

            else:
                # LM
                model_train.fit(X_train, y_train)

            y_test_pr = model_train.predict(X_test)
            y_train_pr = model_train.predict(X_train)

            sc_te = self.eval_method(y_test, y_test_pr)
            sc_tr = self.eval_method(y_train, y_train_pr)

            if self.threshold_score is True:
                if sc_tr < 0:
                    sc_tr = 0
                if sc_te < 0:
                    sc_te = 0

            self.score_train.append(sc_tr)
            self.score_test.append(sc_te)
            self.X_train.append(X_train)
            self.X_test.append(X_test)
            self.y_train.append(y_train)
            self.y_test.append(y_test)
            self.y_train_pr.append(y_train_pr)
            self.y_test_pr.append(y_test_pr)

        return np.mean(self.score_test)

    def run_Bay_Opt(self, space, rounds=10, base_estimator="GP", acq_func="EI",
                    acq_optimizer="sampling", initial_point_generator="lhs"):
        """Run skopt bayesian optimization
        skopt.Optimizer:
        https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer

        example:
        https://scikit-optimize.github.io/stable/auto_examples/ask-and-tell.html#sphx-glr-auto-examples-ask-and-tell-py

        Special attention needs to be made with the run_CV output,
        some metrics are minimized (MAE), some are maximized (r^2)

        Parameters
        ----------
        space : skopt hyperparameter space definition
        rounds : int, optional
            optimizing rounds, by default 10
        base_estimator : str, optional
            surrogate model, used as optimization function instead of cross validation, by default "GP"
        acq_func : str, optional
            function to minimize over the posterior distribution, by default "EI"
        acq_optimizer : str, optional
            method to minimize the acquisition function, by default "sampling"
        initial_point_generator : str, optional
            sets a initial point generator, by default "lhs"

        Returns
        -------
        skopt result
        """

        self.space = space

        opt = Optimizer(self.space, base_estimator=base_estimator, acq_func=acq_func,
                        acq_optimizer=acq_optimizer,
                        initial_point_generator=initial_point_generator)
        for _ in range(rounds):
            next_x = opt.ask()
            # set model values
            for i in range(len(next_x)):
                setattr(self.model, self.space[i].name, next_x[i])
            f_val = self.run_CV()
            res = opt.tell(next_x, f_val)
            print(f_val)

        # res is here automatically appended by skopt
        self.best_metric = res.fun
        self.best_params = res.x
        return res

    def train_final_model(self, bayes_opt=True) -> None:
        """Train final model on all data

        Parameters
        ----------
        bayes_opt : boolean
            if True get best bayesian optimization parameters and train model with the
            according parameters
        """
        if bayes_opt is True:
            for i in range(len(self.best_params)):
                setattr(self.model, self.space[i].name, self.best_params[i])

        self.model.fit(self.data, self.label)

    def save(self, str_save_add=None) -> None:
        """Saves decoder object to pickle
        """

        # run_analysis does not need to be saved twice, since grid points are saved as well
        if self.run_analysis is not None:
            self.run_analysis = None

        if str_save_add is None:
            PATH_OUT = os.path.join(self.feature_path, self.feature_file, self.feature_file + "_ML_RES.p")
        else:
            PATH_OUT = os.path.join(self.feature_path, self.feature_file, self.feature_file +
                                    "_" + str_save_add + "_ML_RES.p")

        print("model being saved to: " + str(PATH_OUT))
        with open(PATH_OUT, 'wb') as output:  # Overwrites any existing file.
            cPickle.dump(self, output)
