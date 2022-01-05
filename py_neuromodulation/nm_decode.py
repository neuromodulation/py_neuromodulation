from numpy.core.overrides import verify_matching_signatures
from sklearn import model_selection, metrics, linear_model, discriminant_analysis, base
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.utils import class_weight
from scipy.ndimage import (binary_dilation,
                           binary_erosion)
from scipy.ndimage import label as label_ndimage
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import os
import json
import numpy as np
import xgboost
import _pickle as cPickle


class Decoder:

    features: pd.DataFrame
    label: np.ndarray
    model: base.BaseEstimator
    cv_method: model_selection.BaseCrossValidator
    threshold_score: bool
    mov_detection_threshold: float
    TRAIN_VAL_SPLIT: bool
    RUN_BAY_OPT: bool
    save_coef: bool
    get_movement_detection_rate: bool
    min_consequent_count: int
    ros: RandomOverSampler = None
    bay_opt_param_space: list = []
    data: np.array
    ch_ind_data: dict
    grid_point_ind_data: dict
    active_gridpoints: list
    feature_names: list[str]
    ch_ind_results: dict = {}
    gridpoint_ind_results: dict = {}
    all_ch_results: dict = {}
    score_train:list = []
    score_test:list = []
    y_test:list = []
    y_train:list = []
    y_test_pr:list = []
    y_train_pr:list = []
    X_test:list = []
    X_train:list = []
    coef:list = []
    mov_detection_rates_test:list = []
    tprate_test:list = []
    fprate_test:list = []
    mov_detection_rates_train:list = []
    tprate_train:list = []
    fprate_train:list = []
    best_bay_opt_params:list = []
    VERBOSE : bool = False

    def __init__(self,
                 features: pd.DataFrame,
                 label: np.ndarray,
                 label_name: str,
                 used_chs: list[str]=None,
                 model=linear_model.LinearRegression(),
                 eval_method=metrics.r2_score,
                 cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                 threshold_score=True,
                 mov_detection_threshold:float =0.5,
                 TRAIN_VAL_SPLIT: bool=True,
                 RUN_BAY_OPT: bool=False,
                 save_coef:bool =False,
                 get_movement_detection_rate:bool =False,
                 min_consequent_count:int =3,
                 bay_opt_param_space: list = [],
                 VERBOSE: bool = False) -> None:
        """Initialize here a feature file for processing
        Read settings.json nm_channels.csv and features.csv
        Read target label

        Parameters
        ----------
        model : machine learning model
            model that utilizes fit and predict functions
        eval_method : sklearn metrics
            evaluation scoring method
        cv_method : sklearm model_selection method
        threshold_score : boolean
            if True set lower threshold at zero (useful for r2),
        mov_detection_threshold : float
            if get_movement_detection_rate is True, find given minimum 'threshold' respective 
            consecutive movement blocks, by default 0.5
        TRAIN_VAL_SPLIT (boolean):
            if true split data into additinal validation, and run class weighted CV
        save_coef (boolean):
            if true, save model._coef trained coefficients
        get_movement_detection_rate (boolean):
            save detection rate and tpr / fpr as well
        min_consequent_count (int):
            if get_movement_detection_rate is True, find given 'min_consequent_count' respective 
            consecutive movement blocks with minimum size of 'min_consequent_count'
        """

        self.features = features
        self.label = label
        self.label_name = label_name
        self.data = np.nan_to_num(np.array(self.features[[col for col in self.features.columns
                                  if not (('time' in col) or (self.label_name in col))]]))

        self.used_chs = used_chs

        self.model = model
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.threshold_score = threshold_score
        self.mov_detection_threshold = mov_detection_threshold
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.RUN_BAY_OPT = RUN_BAY_OPT
        self.save_coef = save_coef
        self.get_movement_detection_rate = get_movement_detection_rate
        self.min_consequent_count = min_consequent_count
        self.bay_opt_param_space = bay_opt_param_space
        self.VERBOSE = VERBOSE

        if type(self.model) is discriminant_analysis.LinearDiscriminantAnalysis:
            self.ros = RandomOverSampler(random_state=0)

    def set_data_ind_channels(self):
        """specified channel individual data
        """
        self.ch_ind_data = {}
        for ch in self.used_chs:
            self.ch_ind_data[ch] = np.nan_to_num(
                np.array(
                    self.features[
                        [col for col in self.features.columns if col.startswith(ch)]
                    ]
                )
            )

    def set_CV_results(self, attr_name, contact_point=None):
        """set CV results in respectie nm_decode attributes
        The reference is first stored in obj_set, and the used lateron

        Parameters
        ----------
        attr_name : string
            is either all_ch_results, ch_ind_results, gridpoint_ind_results
        contact_point : object, optional
            usually an int specifying the grid_point or string, specifying the used channel,
            by default None
        """
        if contact_point is not None:
            getattr(self, attr_name)[contact_point] = {}
            obj_set = getattr(self, attr_name)[contact_point]
        else:
            obj_set = getattr(self, attr_name)

        obj_set["score_train"] = self.score_train
        obj_set["score_test"] = self.score_test
        obj_set["y_test"] = self.y_test
        obj_set["y_train"] = self.y_train
        obj_set["y_test_pr"] = self.y_test_pr
        obj_set["y_train_pr"] = self.y_train_pr
        obj_set["X_train"] = self.X_train
        obj_set["X_test"] = self.X_test
        if self.save_coef:
            obj_set["coef"] = self.coef
        if self.get_movement_detection_rate:
            obj_set["mov_detection_rate_test"] = self.mov_detection_rates_test
            obj_set["mov_detection_rate_train"] = self.mov_detection_rates_train
            obj_set["fprate_test"] = self.fprate_test
            obj_set["fprate_train"] = self.fprate_train
            obj_set["tprate_test"] = self.tprate_test
            obj_set["tprate_train"] = self.tprate_train

    def run_CV_caller(self, feature_contacts: str="ind_channels"):
        """[summary]

        Parameters
        ----------
        feature_contacts : str, optional
            [description], by default "ind_channels"
        """
        valid_feature_contacts = ["ind_channels", "all_channels_combined", "grid_points"]
        if feature_contacts not in valid_feature_contacts:
            raise ValueError(f"{feature_contacts} not in {valid_feature_contacts}")

        if feature_contacts == "grid_points":
            for grid_point in self.active_gridpoints:
                self.run_CV(self.grid_point_ind_data[grid_point], self.label)
                self.set_CV_results('gridpoint_ind_results', contact_point=grid_point)
            return self.gridpoint_ind_results

        if feature_contacts == "ind_channels":
            for ch in self.used_chs:
                self.run_CV(self.ch_ind_data[ch], self.label)
                self.set_CV_results('ch_ind_results', contact_point=ch)
            return self.ch_ind_results

        if feature_contacts == "all_channels_combined":
            dat_combined = np.concatenate(list(self.ch_ind_data.values()), axis=1)
            self.run_CV(dat_combined, self.label)
            self.set_CV_results('all_ch_results', contact_point=None)
            return self.all_ch_results

    def set_data_grid_points(self, cortex_only=False, subcortex_only=False):
        """Read the run_analysis
        Projected data has the shape (samples, grid points, features)
        """

        # activate_gridpoints stores cortex + subcortex data
        self.active_gridpoints = np.unique(
            [i.split('_')[0] + "_" + i.split('_')[1]
            for i in self.features.columns 
                if "grid" in i]
        )

        if cortex_only:
            self.active_gridpoints = [
                i
                for i in self.active_gridpoints 
                if i.startswith("gridcortex")
            ]

        if subcortex_only:
            self.active_gridpoints = [
                i
                for i in self.active_gridpoints 
                if i.startswith("gridsubcortex")
            ]

        self.feature_names = [
            i[len(self.active_gridpoints[0]+"_"):] 
            for i in self.features.columns 
                if self.active_gridpoints[0]+"_" in i
        ]

        self.grid_point_ind_data = {}

        self.grid_point_ind_data = {
            grid_point : np.nan_to_num(self.features[
                    [i 
                    for i in self.features.columns 
                        if grid_point in i]
                    ]
            )
            for grid_point in self.active_gridpoints
        }

    def get_movement_grouped_array(self, prediction, threshold=0.5, min_consequent_count=5):
        """Return given a 1D numpy array, an array of same size with grouped consective blocks

        Parameters
        ----------
        prediction : np.array
            numpy array of either predictions or labels, that is going to be grouped
        threshold : float, optional
            threshold to be applied to 'prediction', by default 0.5
        min_consequent_count : int, optional
            minimum required consective samples higher than 'threshold', by default 5

        Returns
        -------
        labeled_array : np.array
            grouped vector with incrementing number for movement blocks
        labels_count : int
            count of individual movement blocks
        """
        mask = prediction > threshold
        structure = [True] * min_consequent_count  # used for erosion and dilation
        eroded = binary_erosion(mask, structure)
        dilated = binary_dilation(eroded, structure)
        labeled_array, labels_count = label_ndimage(dilated)
        return labeled_array, labels_count

    def calc_movement_detection_rate(self, y_label, prediction, threshold=0.5, min_consequent_count=3):
        """Given a label and prediction, return the movement detection rate on the basis of 
        movements classified in blocks of 'min_consequent_count'.

        Parameters
        ----------
        y_label : [type]
            [description]
        prediction : [type]
            [description]
        threshold : float, optional
            threshold to be applied to 'prediction', by default 0.5
        min_consequent_count : int, optional
            minimum required consective samples higher than 'threshold', by default 3

        Returns
        -------
        mov_detection_rate : float
            movement detection rate, where at least 'min_consequent_count' samples where high in prediction
        fpr : np.array
            sklearn.metrics false positive rate np.array
        tpr : np.array
            sklearn.metrics true positive rate np.array
        """

        pred_grouped, _ = self.get_movement_grouped_array(prediction, threshold, min_consequent_count)
        y_grouped, labels_count = self.get_movement_grouped_array(y_label, threshold, min_consequent_count)

        hit_rate = np.zeros(labels_count)
        pred_group_bin = np.array(pred_grouped>0)

        for label_number in range(1, labels_count + 1):  # labeling starts from 1    
            hit_rate[label_number-1] = np.sum(pred_group_bin[np.where(y_grouped == label_number)[0]])

        try:
            mov_detection_rate = np.where(hit_rate>0)[0].shape[0] / labels_count
        except ZeroDivisionError:
            print("no movements in label")
            return 0, 0, 0

        # calculating TPR and FPR: https://stackoverflow.com/a/40324184/5060208
        CM = metrics.confusion_matrix(y_label, prediction)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

        return mov_detection_rate, fpr, tpr
    
    def init_cv_res(self) -> None:
        self.score_train = []
        self.score_test = []
        self.y_test = []
        self.y_train = []
        self.y_test_pr = []
        self.y_train_pr = []
        self.X_test = []
        self.X_train = []
        self.coef = []
        if self.get_movement_detection_rate is True:
            self.mov_detection_rates_test = []
            self.tprate_test = []
            self.fprate_test = []
            self.mov_detection_rates_train = []
            self.tprate_train = []
            self.fprate_train = []
        if self.RUN_BAY_OPT is True:
            self.best_bay_opt_params = []

    def run_CV(self, data=None, label=None):
        """Evaluate model performance on the specified cross validation.
        If no data and label is specified, use whole feature class attributes.

        Parameters
        ----------
        data (np.ndarray):
            data to train and test with shape samples, features
        label (np.ndarray):
            label to train and test with shape samples, features
        Returns
        -------
        cv_res : float
            mean cross validation result
        """

        self.init_cv_res()

        if data is None:
            print("use all channel data as features")
            data = self.data
            label = self.label

        for train_index, test_index in self.cv_method.split(self.data):

            model_train = clone(self.model)
            X_train, y_train = data[train_index, :], label[train_index]
            X_test, y_test = data[test_index], label[test_index]

            if y_train.sum() == 0:  # only one class present
                continue

            if self.RUN_BAY_OPT is True:

                X_train_bo, X_test_bo, y_train_bo, y_test_bo = \
                    model_selection.train_test_split(
                        X_train, y_train, train_size=0.7, shuffle=False)

                if y_train_bo.sum() == 0 or y_test_bo.sum() == 0:
                    print("could not start Bay. Opt. with no labels > 0")
                    continue

                params_bo = self.run_Bay_Opt(
                    X_train_bo,
                    y_train_bo,
                    X_test_bo,
                    y_test_bo,
                    rounds=10
                )

                # set bay. opt. obtained best params to model
                params_bo_dict = {}
                for i in range(len(params_bo)):
                    setattr(
                        model_train,
                        self.bay_opt_param_space[i].name,
                        params_bo[i]
                    )
                    params_bo_dict[self.bay_opt_param_space[i].name] = params_bo[i]
                self.best_bay_opt_params.append(params_bo_dict)

            # optionally split training data also into train and validation
            if self.TRAIN_VAL_SPLIT is True:
                X_train, X_val, y_train, y_val = \
                    model_selection.train_test_split(
                        X_train, y_train, train_size=0.7, shuffle=False)
                if y_train.sum() == 0:
                    continue

                #classes_weights = class_weight.compute_sample_weight(
                #    class_weight='balanced', y=y_train)

                model_train.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)],
                    early_stopping_rounds=7, #sample_weight=classes_weights,
                    verbose=self.VERBOSE, eval_metric="logloss")
            else:
                # check for LDA; and apply rebalancing
                if type(model_train) is discriminant_analysis.LinearDiscriminantAnalysis:
                    X_train, y_train = self.ros.fit_resample(X_train, y_train)

                if type(model_train) is xgboost.sklearn.XGBClassifier:
                    model_train.fit(X_train, y_train, eval_metric="logloss")  # to avoid warning
                else:
                    model_train.fit(X_train, y_train)

            if self.save_coef:
                self.coef.append(model_train.coef_)

            y_test_pr = model_train.predict(X_test)
            y_train_pr = model_train.predict(X_train)

            sc_te = self.eval_method(y_test, y_test_pr)
            sc_tr = self.eval_method(y_train, y_train_pr)

            if self.threshold_score is True:
                if sc_tr < 0:
                    sc_tr = 0
                if sc_te < 0:
                    sc_te = 0

            if self.get_movement_detection_rate is True:
                mov_detection_rate, fpr, tpr = self.calc_movement_detection_rate(
                    y_test,
                    y_test_pr,
                    self.mov_detection_threshold,
                    self.min_consequent_count
                )

                self.mov_detection_rates_test.append(mov_detection_rate)
                self.tprate_test.append(tpr)
                self.fprate_test.append(fpr)

                mov_detection_rate, fpr, tpr = self.calc_movement_detection_rate(
                    y_train,
                    y_train_pr,
                    self.mov_detection_threshold,
                    self.min_consequent_count
                )

                self.mov_detection_rates_train.append(mov_detection_rate)
                self.tprate_train.append(tpr)
                self.fprate_train.append(fpr)

            self.score_train.append(sc_tr)
            self.score_test.append(sc_te)
            self.X_train.append(X_train)
            self.X_test.append(X_test)
            self.y_train.append(y_train)
            self.y_test.append(y_test)
            self.y_train_pr.append(y_train_pr)
            self.y_test_pr.append(y_test_pr)

        return np.mean(self.score_test)

    def run_Bay_Opt(self,
        X_train,
        y_train,
        X_test,
        y_test,
        rounds=10,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        initial_point_generator="lhs"
    ):
        """Run skopt bayesian optimization
        skopt.Optimizer:
        https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer

        example:
        https://scikit-optimize.github.io/stable/auto_examples/ask-and-tell.html#sphx-glr-auto-examples-ask-and-tell-py

        Special attention needs to be made with the run_CV output,
        some metrics are minimized (MAE), some are maximized (r^2)

        Parameters
        ----------
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
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
        skopt result parameters
        """

        def get_f_val(model_bo):
            if type(model_bo) is xgboost.sklearn.XGBClassifier:
                # for avoiding xgboost warning
                model_bo.fit(X_train, y_train, eval_metric="logloss")
            else:
                model_bo.fit(X_train, y_train)
            return self.eval_method(y_test, model_bo.predict(X_test))

        opt = Optimizer(
            self.bay_opt_param_space,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            initial_point_generator=initial_point_generator
        )
        for _ in range(rounds):
            next_x = opt.ask()
            # set model values
            model_bo = clone(self.model)
            for i in range(len(next_x)):
                setattr(model_bo, self.bay_opt_param_space[i].name, next_x[i])
            f_val = get_f_val(model_bo)
            res = opt.tell(next_x, f_val)
            if self.VERBOSE:
                print(f_val)

        # res is here automatically appended by skopt
        return res.x

    def save(self, feature_path: str, feature_file: str, str_save_add=None) -> None:
        """Save decoder object to pickle
        """

        # why is the decoder not saved to a .json?

        if str_save_add is None:
            PATH_OUT = os.path.join(feature_path, feature_file, feature_file + "_ML_RES.p")
        else:
            PATH_OUT = os.path.join(feature_path, feature_file, feature_file +
                                    "_" + str_save_add + "_ML_RES.p")

        print("model being saved to: " + str(PATH_OUT))
        with open(PATH_OUT, 'wb') as output:  # Overwrites any existing file.
            cPickle.dump(self, output)
