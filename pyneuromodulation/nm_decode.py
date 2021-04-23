# read feature file 
# setup 3 fold non shuffled CV 
# what woult be optimal
# initialize with a single run, extract features; and label 
# how do I know which one is the label? 

from sklearn import model_selection, metrics, linear_model
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer
from sklearn.linear_model import ElasticNet
from sklearn.base import clone

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
        Read settings.json df_M1.tsv and features.csv
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

        self.df_M1 = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                 feature_file + "_DF_M1.csv"), header=0)

        with open(os.path.join(self.feature_path, feature_file,
                               feature_file + "_SETTINGS.json")) as f:
            self.settings = json.load(f)

        # for simplicity, choose here only first one
        # lateron check laterality
        self.target_ch = self.df_M1[self.df_M1["target"] == 1]["name"].iloc[0]

        self.label = np.nan_to_num(np.array(self.features[self.target_ch]))
        self.data = np.nan_to_num(np.array(self.features[[col for col in self.features.columns
                                  if not (('time' in col) or (self.target_ch in col))]]))

        # crop here features for example
        self.data = self.data[:, :100]

        self.model = model
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.threshold_score = threshold_score

    def run_CV(self):
        """Evaluate model performance on the specified cross validation

        Parameters
        ----------

        Returns
        -------
        cv_res : float
            mean cross validation result
        """

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
            X_train, y_train = self.data[train_index, :], self.label[train_index]
            X_test, y_test = self.data[test_index], self.label[test_index]

            # optionally split training data also into train and validation
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8,shuffle=False)

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

    def save(self) -> None:
        """Saves decoder object to pickle
        """
        PATH_OUT = os.path.join(self.feature_path, self.feature_file, self.feature_file + "_ML_RES.p")
        print("model being saved to: " + str(PATH_OUT))
        with open(PATH_OUT, 'wb') as output:  # Overwrites any existing file.
            cPickle.dump(self, output)
