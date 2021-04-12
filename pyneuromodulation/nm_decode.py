# read feature file 
# setup 3 fold non shuffled CV 
# what woult be optimal
# initialize with a single run, extract features; and label 
# how do I know which one is the label? 

from sklearn import model_selection, metrics, linear_model
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.linear_model import ElasticNet

import pandas as pd
import os
import json
import numpy as np
import xgboost as xgb


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
                                    feature_file+"_FEATURES.csv"), header=0)

        self.df_M1 = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                 feature_file+"_DF_M1.csv"), header=0)

        with open(os.path.join(self.feature_path, feature_file,
                               feature_file+"_SETTINGS.json")) as f:
            self.settings = json.load(f)

        # for simplicity, choose here only first one
        # lateron check laterality
        self.target_ch = self.df_M1[self.df_M1["target"] == 1]["name"].iloc[0]

        label = np.nan_to_num(np.array(self.features[self.target_ch]))
        data = np.nan_to_num(np.array(self.features[[col for col in self.features.columns 
                                  if not (('time' in col) or (self.target_ch in col))]]))
        
        # limit number of features for testing
        data = self.data[:, :100]
        
        self.model = model
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.threshold_score = threshold_score

    def init_bayesian_opt(self, space_used=None) -> None:
        """Used skopt hyperparameter space

        Parameters
        ----------
        space_used : list, optional
            list of bayesian optimization hyperparameters to optimize, by default None
        """
        self.space_LM = [Real(0, 1, "uniform", name='alpha'),
                         Real(0, 1, "uniform", name='l1_ratio')]

        self.space_XGB = [Integer(1, 100, name='max_depth'),
                          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
                          Real(10**0, 10**1, "uniform", name="gamma")]

        if space_used is None:
            self.space_used = self.space_LM
        else: 
            self.space_used = space_used
        
        bay_opt_space = space_used

    '''
    def optimize_enet(self, x, y, calls=10, scoring_method='r2', cv_runs=3) -> np.ndarray:
        """Run bayesian optimization for elastic net linar model

        Parameters
        ----------
        x : np.ndarray
            features
        y : np.ndarray
            labels
        calls : int, optional
            bayesian optimization rounds, by default 10

        Returns
        -------
        np.ndarray
            bayesian optimization retults
        """
        @use_named_args(self.space_LM)
        def objective(**params):
            reg=ElasticNet(max_iter=1000, normalize=False)
            reg.set_params(**params)
            cval = model_selection.cross_val_score(reg, x, y, scoring='r2', cv=3)
            cval[np.where(cval < 0)[0]] = 0

            return -cval.mean()

        res_gp = gp_minimize(objective, self.space_LM, n_calls=20, random_state=0)
        return res_gp

    def optimize_xgb(self, x, y, calls=10, scoring_method='r2', cv_runs=3):

        def r2_evalerror(preds, dtrain):
            """
            Return R^2 evalerror, since xgb doesn't support r2 by default
            Return's pair metric inlcuding metric name for skopt requirements

            Parameters
            ----------
            preds : np.ndarray
                prediction array
            dtrain : xgb.data

            Returns
            -------
            eval_name : string
                name of used metric
            eval_value : float
            """

            labels = dtrain.get_label()

            r2 = metrics.r2_score(labels, preds)

            if r2 < 0:
                r2 = 0

            return 'r2', r2

        @use_named_args(self.space_XGB)
        def objective(self, **params):
            print(params)

            params_ = {'max_depth': int(params["max_depth"]),
                'gamma': params['gamma'],
                # 'n_estimators': int(params["n_estimators"]),
                'learning_rate': params["learning_rate"],
                'subsample': 0.8,
                'eta': 0.1,
                'disable_default_eval_metric' : 1}
                # 'scale_pos_weight ' : 1}
                # 'nthread':59}
                # 'tree_method' : 'gpu_hist'}
                # 'gpu_id' : 1}

            cv_result = xgb.cv(params_, xgb.DMatrix(x, label=y), num_boost_round=calls, 
                               feval=r2_evalerror, nfold=3)
            return -cv_result['test-r2-mean'].iloc[-1]

        res_gp = gp_minimize(objective, self.space_XGB, n_calls=20, random_state=0)
        return res_gp
        
    '''

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

            model_train = self.model
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

        return -np.mean(self.score_test)

    def run_CV_Bay_Opt(**params):
        """Evaluate model performance on the specified cross validation

        Parameters
        ----------

        Returns
        -------
        cv_res : float
            mean cross validation result
        """

        # if linear model being used
        if bay_opt is True:
            self.model.set_params(**params)

        # if xgboost being used, might be necessary to set the params individually

        score_train = []
        score_test = []
        y_test = []
        y_train = []
        X_test = []
        X_train = []

        for train_index, test_index in cv_method.split(data):
            
            model_train = model 

            X_train, X_test = data[train_index, :], label[test_index]
            y_train, y_test = data[train_index], label[test_index]

            # optionally split training data also into train and validation
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8,shuffle=False)

            model_train.fit(X_train, y_train)

            y_test_pr = model_train.predict(X_test)
            y_train_pr = model_train.predict(X_train)

            sc_te = eval_method(y_test, y_test_pr)
            sc_tr = eval_method(y_train, y_train_pr)

            if threshold_score is True:
                if sc_tr < 0:
                    sc_tr = 0
                if sc_te < 0:
                    sc_te = 0

            score_train.append(sc_tr)
            score_test.append(sc_te)
            X_train.append(X_train)
            X_test.append(X_test)
            y_train.append(y_train)
            y_test.append(y_test)

        return -np.mean(score_test)

    def bay_opt(self, n_calls=20):

        res_gp = gp_minimize(run_CV, self.space_used, n_calls=n_calls, random_state=0)
        return res_gp