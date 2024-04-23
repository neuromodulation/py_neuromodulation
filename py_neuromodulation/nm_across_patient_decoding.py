import numpy as np
import os
import pandas as pd
from sklearn import metrics, linear_model, model_selection

import py_neuromodulation
from py_neuromodulation import nm_decode, nm_RMAP


class AcrossPatientRunner:
    def __init__(
        self,
        outpath: str,
        model=linear_model.LogisticRegression(class_weight="balanced"),
        TRAIN_VAL_SPLIT=False,
        eval_method=metrics.balanced_accuracy_score,
        cv_method=model_selection.KFold(n_splits=3, shuffle=False),
        VERBOSE=False,
        use_nested_cv=True,
        RUN_BAY_OPT=False,
        ML_model_name="LM",
        cohorts: list | None = None,
        load_channel_all: bool = False,
        load_grid_point_all: bool = False,
    ) -> None:

        self.outpath = outpath
        self.model = model
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.use_nested_cv = use_nested_cv
        self.RUN_BAY_OPT = RUN_BAY_OPT
        self.VERBOSE = VERBOSE
        self.eval_method = eval_method
        self.ML_model_name = ML_model_name
        self.cv_method = cv_method
        self.cohorts = cohorts

        self.grid_cortex = pd.read_csv(
            os.path.join(py_neuromodulation.__path__[0], "grid_cortex.tsv"),
            sep="\t",
        ).to_numpy()

        self.RMAPSelector = nm_RMAP.RMAPChannelSelector()

        if load_channel_all is True:
            self.ch_all = np.load(
                os.path.join(self.outpath, "channel_all.npy"),
                allow_pickle="TRUE",
            ).item()
        if load_grid_point_all is True:
            self.grid_point_all = np.load(
                os.path.join(self.outpath, "grid_point_all.npy"),
                allow_pickle="TRUE",
            ).item()

    def init_decoder(self) -> nm_decode.Decoder:

        return nm_decode.Decoder(
            model=self.model,
            TRAIN_VAL_SPLIT=self.TRAIN_VAL_SPLIT,
            get_movement_detection_rate=True,
            eval_method=self.eval_method,
            VERBOSE=self.VERBOSE,
            cv_method=self.cv_method,
            use_nested_cv=self.use_nested_cv,
            RUN_BAY_OPT=self.RUN_BAY_OPT,
        )

    def eval_model(self, X_train, y_train, X_test, y_test):

        return self.decoder.wrapper_model_train(
            X_train,
            y_train,
            X_test,
            y_test,
            cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
        )

    @staticmethod
    def get_data_sub_ch(channel_all, cohort, sub, ch):

        X_train = []
        y_train = []

        for f in channel_all[cohort][sub][ch].keys():
            X_train.append(channel_all[cohort][sub][ch][f]["data"])
            y_train.append(channel_all[cohort][sub][ch][f]["label"])
        if len(X_train) > 1:
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
        else:
            X_train = X_train[0]
            y_train = y_train[0]

        return X_train, y_train

    def get_patients_train_dict(self, sub_test, cohort_test, val_approach: str, data_select: dict):
        cohorts_train = {}
        for cohort in self.cohorts:
            if val_approach == "leave_1_cohort_out" and cohort == cohort_test:
                continue
            if (
                val_approach == "leave_1_sub_out_within_coh"
                and cohort != cohort_test
            ):
                continue
            cohorts_train[cohort] = []
            for sub in data_select[cohort]:
                if (
                    val_approach == "leave_1_sub_out_within_coh"
                    and sub == sub_test
                    and cohort == cohort_test
                ):
                    continue
                if (
                    val_approach == "leave_1_sub_out_across_coh"
                    and sub == sub_test
                ):
                    continue
                cohorts_train[cohort].append(sub)
        return cohorts_train

    def get_data_grid_point(
        self, sub_test: str, cohort_test: str, best_gp_list: list
    ):
        for gp in best_gp_list:
            if gp in self.grid_point_all[cohort_test][sub_test]:
                X_test, y_test = self.get_data_sub_ch(
                    self.grid_point_all, cohort_test, sub_test, gp
                )
                break
            else:
                continue
        return X_test, y_test

    def get_data_channels(self, sub_test: str, cohort_test: str, df_rmap: list):
        ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
            "ch"
        ].iloc[0]
        X_test, y_test = self.get_data_sub_ch(
            self.ch_all, cohort_test, sub_test, ch_test
        )
        return X_test, y_test

    def cross_val_approach_RMAP(
        self,
        val_approach: str = "leave_1_cohort_out",
        df_select: pd.DataFrame | None = None,
        select_best_gp: bool = False,
        add_UPDRS: bool = False,
        df_updrs: pd.DataFrame | None = None,
    ):

        if select_best_gp is True:
            best_gp_list = list(
                df_select.sort_values("performance_test", ascending=False)["ch"]
            )
            data_select = self.grid_point_all
        else:
            data_select = self.ch_all

        p_ = {}
        for cohort_test in self.cohorts:
            print(cohort_test)
            if cohort_test not in p_:
                p_[cohort_test] = {}
            for sub_test in data_select[cohort_test].keys():
                print(sub_test)
                if sub_test not in p_[cohort_test]:
                    p_[cohort_test][sub_test] = {}
                if select_best_gp is True:
                    X_test, y_test = self.get_data_grid_point(
                        sub_test, cohort_test, best_gp_list
                    )
                else:
                    X_test, y_test = self.get_data_channels(
                        sub_test, cohort_test, df_rmap=df_select
                    )

                if add_UPDRS is True:
                    updrs = df_updrs.query(
                        "sub == @sub_test and cohort == @cohort_test"
                    ).iloc[0]["UPDRS_total"]
                    if np.isnan(updrs):
                        continue
                    X_test = np.concatenate(
                        (
                            X_test,
                            np.expand_dims(
                                np.repeat(updrs, repeats=X_test.shape[0]),
                                axis=1,
                            ),
                        ),
                        axis=1,
                    )

                cohorts_train = self.get_patients_train_dict(
                    sub_test, cohort_test, val_approach=val_approach, data_select=data_select
                )

                X_train_comb = []
                y_train_comb = []
                for cohort_train in list(cohorts_train.keys()):
                    for sub_train in cohorts_train[cohort_train]:
                        if select_best_gp is True:
                            X_train, y_train = self.get_data_grid_point(
                                sub_train, cohort_train, best_gp_list
                            )
                        else:
                            X_train, y_train = self.get_data_channels(
                                sub_train, cohort_train, df_rmap=df_select
                            )
                        if add_UPDRS is True:
                            updrs = df_updrs.query(
                                "sub == @sub_train and cohort == @cohort_train"
                            ).iloc[0]["UPDRS_total"]
                            if np.isnan(
                                updrs
                            ):  # the returned True is here not boolean but np boolean
                                continue
                            X_train = np.concatenate(
                                (
                                    X_train,
                                    np.expand_dims(
                                        np.repeat(
                                            updrs, repeats=X_train.shape[0]
                                        ),
                                        axis=1,
                                    ),
                                ),
                                axis=1,
                            )

                        X_train_comb.append(X_train)
                        y_train_comb.append(y_train)
                if len(X_train_comb) > 1:
                    X_train = np.concatenate(X_train_comb, axis=0)
                    y_train = np.concatenate(y_train_comb, axis=0)
                else:
                    X_train = X_train_comb[0]
                    y_train = X_train_comb[0]

                self.decoder = self.init_decoder()

                # X_train, y_train, X_test, y_test = self.decoder.append_samples_val(X_train, y_train, X_test, y_test, 5)

                model = self.decoder.wrapper_model_train(
                    X_train=X_train,
                    y_train=y_train,
                    return_fitted_model_only=True,
                )
                cv_res = self.decoder.eval_model(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
                    save_data=False,
                )
                p_[cohort_test][sub_test] = cv_res
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name + f"_performance_{val_approach}_RMAP.npy",
            ),
            p_,
        )

    def cross_val_p2p_RMAP(
        self,
        df_select: pd.DataFrame | None = None,
        select_best_gp: bool = False,
    ):
        if select_best_gp is True:
            best_gp_list = list(
                df_select.sort_values("performance_test", ascending=False)["ch"]
            )
            data_select = self.grid_point_all
        else:
            data_select = self.ch_all

        p_ = {}
        for cohort_test in self.cohorts:
            print(cohort_test)
            if cohort_test not in p_:
                p_[cohort_test] = {}
            for sub_test in data_select[cohort_test].keys():
                print(sub_test)
                if sub_test not in p_[cohort_test]:
                    p_[cohort_test][sub_test] = {}

                if select_best_gp is True:
                    X_test, y_test = self.get_data_grid_point(
                        sub_test, cohort_test, best_gp_list
                    )
                else:
                    X_test, y_test = self.get_data_channels(
                        sub_test, cohort_test, df_rmap=df_select
                    )
                for cohort_train in self.cohorts:
                    if cohort_train not in p_[cohort_test][sub_test]:
                        p_[cohort_test][sub_test][cohort_train] = {}
                    for sub_train in list(data_select[cohort_train].keys()):
                        if (
                            sub_train
                            not in p_[cohort_test][sub_test][cohort_train]
                        ):
                            p_[cohort_test][sub_test][cohort_train][
                                sub_train
                            ] = {}

                        if select_best_gp is True:
                            X_train, y_train = self.get_data_grid_point(
                                sub_train, cohort_train, best_gp_list
                            )
                        else:
                            X_train, y_train = self.get_data_channels(
                                sub_train, cohort_train, df_rmap=df_select
                            )
                        self.decoder = self.init_decoder()

                        # X_train, y_train, X_test, y_test = self.decoder.append_samples_val(X_train, y_train, X_test, y_test, 5)

                        model = self.decoder.wrapper_model_train(
                            X_train=X_train,
                            y_train=y_train,
                            return_fitted_model_only=True,
                        )
                        cv_res = self.decoder.eval_model(
                            model,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            cv_res=nm_decode.CV_res(
                                get_movement_detection_rate=True
                            ),
                            save_data=False,
                        )
                        p_[cohort_test][sub_test][cohort_train][
                            sub_train
                        ] = cv_res

        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name + "_performance_p2p_RMAP.npy",
            ),
            p_,
        )

    def leave_one_patient_out_RMAP(
        self, val_approach: str = "leave_1_cohort_out"
    ):
        p_ = {}
        for cohort_test in self.cohorts:
            if cohort_test not in p_:
                p_[cohort_test] = {}
            for sub_test in self.ch_all[cohort_test].keys():
                if sub_test not in p_:
                    p_[cohort_test][sub_test] = {}
                for ch_test in self.ch_all[cohort_test][sub_test].keys():
                    if ch_test not in p_[cohort_test][sub_test]:
                        p_[cohort_test][sub_test][ch_test] = {}

                    cohorts_train = self.get_patients_train_dict(
                        sub_test, cohort_test, val_approach=val_approach
                    )

                    (
                        cohort_train,
                        sub_train,
                        ch_train,
                    ) = self.RMAPSelector.get_highest_corr_sub_ch(
                        cohort_test,
                        sub_test,
                        ch_test,
                        cohorts_train,
                        path_dir=r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity",
                    )

                    X_train, y_train = self.get_data_sub_ch(
                        self.ch_all, cohort_train, sub_train, ch_train
                    )
                    X_test, y_test = self.get_data_sub_ch(
                        self.ch_all, cohort_test, sub_test, ch_test
                    )

                    self.decoder = self.init_decoder()

                    model = self.decoder.wrapper_model_train(
                        X_train=X_train,
                        y_train=y_train,
                        return_fitted_model_only=True,
                    )
                    cv_res = self.decoder.eval_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        cv_res=nm_decode.CV_res(
                            get_movement_detection_rate=True
                        ),
                        save_data=False,
                        append_samples=True,
                    )
                    p_[cohort_test][sub_test][ch_test] = cv_res
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name
                + "_performance_leave_one_cohort_out_RMAP.npy",
            ),
            p_,
        )

    def run_cohort_leave_one_patient_out_CV_within_cohort(self):

        grid_point_all = np.load(
            os.path.join(self.outpath, "grid_point_all.npy"),
            allow_pickle="TRUE",
        ).item()
        performance_leave_one_patient_out = {}

        for cohort in self.cohorts:
            print("cohort: " + str(cohort))
            performance_leave_one_patient_out[cohort] = {}

            for grid_point in list(grid_point_all.keys()):
                print("grid point: " + str(grid_point))
                if cohort not in grid_point_all[grid_point]:
                    continue
                if len(list(grid_point_all[grid_point][cohort].keys())) <= 1:
                    continue  # cannot do leave one out prediction with a single subject
                performance_leave_one_patient_out[cohort][grid_point] = {}

                for subject_test in list(
                    grid_point_all[grid_point][cohort].keys()
                ):
                    X_test = []
                    y_test = []
                    for run in list(
                        grid_point_all[grid_point][cohort][subject_test].keys()
                    ):
                        if (
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["lat"]
                            != "CON"
                        ):
                            continue
                        X_test.append(
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["data"]
                        )
                        y_test.append(
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["label"]
                        )
                    if len(X_test) > 1:
                        X_test = np.concatenate(X_test, axis=0)
                        y_test = np.concatenate(y_test, axis=0)
                    else:
                        X_test = X_test[0]
                        y_test = y_test[0]
                    X_train = []
                    y_train = []
                    for subject_train in list(
                        grid_point_all[grid_point][cohort].keys()
                    ):
                        if subject_test == subject_train:
                            continue
                        for run in list(
                            grid_point_all[grid_point][cohort][
                                subject_train
                            ].keys()
                        ):
                            if (
                                grid_point_all[grid_point][cohort][
                                    subject_train
                                ][run]["lat"]
                                != "CON"
                            ):
                                continue
                            X_train.append(
                                grid_point_all[grid_point][cohort][
                                    subject_train
                                ][run]["data"]
                            )
                            y_train.append(
                                grid_point_all[grid_point][cohort][
                                    subject_train
                                ][run]["label"]
                            )
                    if len(X_train) > 1:
                        X_train = np.concatenate(X_train, axis=0)
                        y_train = np.concatenate(y_train, axis=0)
                    else:
                        X_train = X_train[0]
                        y_train = y_train[0]

                    # run here ML estimation
                    self.decoder = self.init_decoder()
                    # Unused variable, remove?
                    #model = self.decoder.wrapper_model_train(
                    #    X_train=X_train,
                    #    y_train=y_train,
                    #    return_fitted_model_only=True,
                    #)
                    # use initialized decoder
                    try:
                        cv_res = self.eval_model(
                            X_train, y_train, X_test, y_test
                        )
                    except nm_decode.Decoder.ClassMissingException:
                        continue

                    performance_leave_one_patient_out[cohort][grid_point][
                        subject_test
                    ] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name
                + "_performance_leave_one_patient_out_within_cohort.npy",
            ),
            performance_leave_one_patient_out,
        )
        return performance_leave_one_patient_out

    def run_cohort_leave_one_cohort_out_CV(self):
        grid_point_all = np.load(
            os.path.join(self.outpath, "grid_point_all.npy"),
            allow_pickle="TRUE",
        ).item()
        performance_leave_one_cohort_out = {}

        for cohort_test in self.cohorts:
            print("cohort: " + str(cohort_test))
            if cohort_test not in performance_leave_one_cohort_out:
                performance_leave_one_cohort_out[cohort_test] = {}

            for grid_point in list(grid_point_all.keys()):
                print("grid point: " + str(grid_point))
                if cohort_test not in grid_point_all[grid_point]:
                    continue
                if len(list(grid_point_all[grid_point].keys())) == 1:
                    continue  # cannot do leave one cohort prediction with a single cohort

                X_train = []
                y_train = []
                for cohort_train in self.cohorts:
                    if cohort_test == cohort_train:
                        continue
                    if cohort_train not in grid_point_all[grid_point]:
                        continue
                    for subject_test in list(
                        grid_point_all[grid_point][cohort_train].keys()
                    ):
                        for run in list(
                            grid_point_all[grid_point][cohort_train][
                                subject_test
                            ].keys()
                        ):
                            if (
                                grid_point_all[grid_point][cohort_train][
                                    subject_test
                                ][run]["lat"]
                                != "CON"
                            ):
                                continue
                            X_train.append(
                                grid_point_all[grid_point][cohort_train][
                                    subject_test
                                ][run]["data"]
                            )
                            y_train.append(
                                grid_point_all[grid_point][cohort_train][
                                    subject_test
                                ][run]["label"]
                            )
                if len(X_train) > 1:
                    X_train = np.concatenate(X_train, axis=0)
                    y_train = np.concatenate(y_train, axis=0)
                else:
                    X_train = X_train[0]
                    y_train = y_train[0]

                # run here ML estimation
                self.decoder = self.init_decoder()
                model = self.decoder.wrapper_model_train(
                    X_train=X_train,
                    y_train=y_train,
                    return_fitted_model_only=True,
                )

                performance_leave_one_cohort_out[cohort_test][grid_point] = {}
                for subject_test in list(
                    grid_point_all[grid_point][cohort_test].keys()
                ):
                    X_test = []
                    y_test = []
                    for run in list(
                        grid_point_all[grid_point][cohort_test][
                            subject_test
                        ].keys()
                    ):
                        if (
                            grid_point_all[grid_point][cohort_test][
                                subject_test
                            ][run]["lat"]
                            != "CON"
                        ):
                            continue
                        X_test.append(
                            grid_point_all[grid_point][cohort_test][
                                subject_test
                            ][run]["data"]
                        )
                        y_test.append(
                            grid_point_all[grid_point][cohort_test][
                                subject_test
                            ][run]["label"]
                        )
                    if len(X_test) > 1:
                        X_test = np.concatenate(X_test, axis=0)
                        y_test = np.concatenate(y_test, axis=0)
                    else:
                        X_test = X_test[0]
                        y_test = y_test[0]

                    cv_res = self.decoder.eval_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        cv_res=nm_decode.CV_res(),
                    )

                    performance_leave_one_cohort_out[cohort_test][grid_point][
                        subject_test
                    ] = cv_res

        performance_leave_one_cohort_out["grid_cortex"] = self.grid_cortex
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name + "_performance_leave_one_cohort_out.npy",
            ),
            performance_leave_one_cohort_out,
        )

    def run_leave_one_patient_out_across_cohorts(self):

        grid_point_all = np.load(
            os.path.join(self.outpath, "grid_point_all.npy"),
            allow_pickle="TRUE",
        ).item()
        performance_leave_one_patient_out = {}

        for grid_point in list(grid_point_all.keys()):
            print("grid point: " + str(grid_point))
            for cohort in self.cohorts:
                print("cohort: " + str(cohort))
                if cohort not in performance_leave_one_patient_out:
                    performance_leave_one_patient_out[cohort] = {}

                if cohort not in grid_point_all[grid_point]:
                    continue
                if len(list(grid_point_all[grid_point][cohort].keys())) <= 1:
                    continue  # cannot do leave one out prediction with a single subject

                if grid_point not in performance_leave_one_patient_out[cohort]:
                    performance_leave_one_patient_out[cohort][grid_point] = {}

                for subject_test in list(
                    grid_point_all[grid_point][cohort].keys()
                ):
                    X_test = []
                    y_test = []
                    for run in list(
                        grid_point_all[grid_point][cohort][subject_test].keys()
                    ):
                        if (
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["lat"]
                            != "CON"
                        ):
                            continue
                        X_test.append(
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["data"]
                        )
                        y_test.append(
                            grid_point_all[grid_point][cohort][subject_test][
                                run
                            ]["label"]
                        )
                    if len(X_test) > 1:
                        X_test = np.concatenate(X_test, axis=0)
                        y_test = np.concatenate(y_test, axis=0)
                    else:
                        X_test = X_test[0]
                        y_test = y_test[0]
                    X_train = []
                    y_train = []
                    for cohort_inner in list(
                        grid_point_all[grid_point].keys()
                    ):  # available cohorts for that grid point
                        for subject_train in list(
                            grid_point_all[grid_point][cohort_inner].keys()
                        ):
                            if (subject_test == subject_train) and (
                                cohort_inner == cohort
                            ):
                                continue
                            for run in list(
                                grid_point_all[grid_point][cohort_inner][
                                    subject_train
                                ].keys()
                            ):
                                if (
                                    grid_point_all[grid_point][cohort_inner][
                                        subject_train
                                    ][run]["lat"]
                                    != "CON"
                                ):
                                    continue
                                X_train.append(
                                    grid_point_all[grid_point][cohort_inner][
                                        subject_train
                                    ][run]["data"]
                                )
                                y_train.append(
                                    grid_point_all[grid_point][cohort_inner][
                                        subject_train
                                    ][run]["label"]
                                )
                    if len(X_train) > 1:
                        X_train = np.concatenate(X_train, axis=0)
                        y_train = np.concatenate(y_train, axis=0)
                    else:
                        X_train = X_train[0]
                        y_train = y_train[0]

                    self.decoder = self.init_decoder()
                    try:
                        cv_res = self.eval_model(
                            X_train, y_train, X_test, y_test
                        )
                    except nm_decode.Decoder.ClassMissingException:
                        continue

                    performance_leave_one_patient_out[cohort][grid_point][
                        subject_test
                    ] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name
                + "_performance_leave_one_patient_out_across_cohorts.npy",
            ),
            performance_leave_one_patient_out,
        )

    def run_leave_nminus1_patient_out_across_cohorts(self):

        grid_point_all = np.load(
            os.path.join(self.outpath, "grid_point_all_re.npy"),
            allow_pickle="TRUE",
        ).item()
        performance_leave_one_patient_out = {}

        for grid_point in list(grid_point_all.keys()):
            print("grid point: " + str(grid_point))
            for cohort_train in self.cohorts:
                print("cohort: " + str(cohort_train))
                if cohort_train not in performance_leave_one_patient_out:
                    performance_leave_one_patient_out[cohort_train] = {}

                if cohort_train not in grid_point_all[grid_point]:
                    continue
                if (
                    len(list(grid_point_all[grid_point][cohort_train].keys()))
                    <= 1
                ):
                    continue  # cannot do leave one out prediction with a single subject
                if (
                    grid_point
                    not in performance_leave_one_patient_out[cohort_train]
                ):
                    performance_leave_one_patient_out[cohort_train][
                        grid_point
                    ] = {}

                for subject_train in list(
                    grid_point_all[grid_point][cohort_train].keys()
                ):
                    X_train = []
                    y_train = []
                    for run in list(
                        grid_point_all[grid_point][cohort_train][
                            subject_train
                        ].keys()
                    ):
                        if (
                            grid_point_all[grid_point][cohort_train][
                                subject_train
                            ][run]["lat"]
                            != "CON"
                        ):
                            continue
                        X_train.append(
                            grid_point_all[grid_point][cohort_train][
                                subject_train
                            ][run]["data"]
                        )
                        y_train.append(
                            grid_point_all[grid_point][cohort_train][
                                subject_train
                            ][run]["label"]
                        )
                    if len(X_train) > 1:
                        X_train = np.concatenate(X_train, axis=0)
                        y_train = np.concatenate(y_train, axis=0)
                    else:
                        X_train = X_train[0]
                        y_train = y_train[0]

                    for cohort_test in list(grid_point_all[grid_point].keys()):
                        for subject_test in list(
                            grid_point_all[grid_point][cohort_test].keys()
                        ):
                            if (subject_test == subject_train) and (
                                cohort_test == cohort_train
                            ):
                                continue
                            X_test = []
                            y_test = []
                            for run in list(
                                grid_point_all[grid_point][cohort_test][
                                    subject_test
                                ].keys()
                            ):
                                if (
                                    grid_point_all[grid_point][cohort_test][
                                        subject_test
                                    ][run]["lat"]
                                    != "CON"
                                ):
                                    continue
                                X_test.append(
                                    grid_point_all[grid_point][cohort_test][
                                        subject_test
                                    ][run]["data"]
                                )
                                y_test.append(
                                    grid_point_all[grid_point][cohort_test][
                                        subject_test
                                    ][run]["label"]
                                )
                            if len(X_test) > 1:
                                X_test = np.concatenate(X_test, axis=0)
                                y_test = np.concatenate(y_test, axis=0)
                            else:
                                X_test = X_test[0]
                                y_test = y_test[0]

                            self.decoder = self.init_decoder()
                            try:
                                cv_res = self.eval_model(
                                    X_train, y_train, X_test, y_test
                                )
                            except nm_decode.Decoder.ClassMissingException:
                                continue

                            if (
                                subject_train
                                not in performance_leave_one_patient_out[
                                    cohort_train
                                ][grid_point]
                            ):
                                performance_leave_one_patient_out[cohort_train][
                                    grid_point
                                ][subject_train] = {}
                            if (
                                cohort_test
                                not in performance_leave_one_patient_out[
                                    cohort_train
                                ][grid_point][subject_train]
                            ):
                                performance_leave_one_patient_out[cohort_train][
                                    grid_point
                                ][subject_train][cohort_test] = {}
                            if (
                                subject_test
                                not in performance_leave_one_patient_out[
                                    cohort_train
                                ][grid_point][subject_train][cohort_test]
                            ):
                                performance_leave_one_patient_out[cohort_train][
                                    grid_point
                                ][subject_train][cohort_test][subject_test] = {}

                            performance_leave_one_patient_out[cohort_train][
                                grid_point
                            ][subject_train][cohort_test][subject_test] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name
                + "_performance_leave_nminus1_patient_out_across_cohorts.npy",
            ),
            performance_leave_one_patient_out,
        )
