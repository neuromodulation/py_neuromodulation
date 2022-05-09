import os
from sklearn import linear_model, metrics, model_selection, discriminant_analysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle

from skopt import space as skopt_space
import xgboost
from py_neuromodulation import nm_GenericStream, nm_analysis, nm_decode


def main():

    res_dict = {}
    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\features_epochs_nonorm"
    for RUN_NAME in [
        "effspm8_JUN_EMO",
        "effspm8_KOR_EMO",
        "effspm8_MIC_EMO",
        "effspm8_NIL_EMO",
        "effspm8_OHL_EMO",
        "effspm8_SCH_EMO",
        "effspm8_THI_EMO",
        "effspm8_WES_EMO",
    ]:
        res_dict[RUN_NAME] = {}
        feature_reader = nm_analysis.Feature_Reader(
            feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
        )

        model = linear_model.LogisticRegression()
        # model = xgboost.XGBClassifier()
        feature_reader.decoder = nm_decode.Decoder(
            features=feature_reader.feature_arr,
            label=feature_reader.label,
            label_name=feature_reader.label_name,
            used_chs=feature_reader.used_chs,
            model=model,
            eval_method=metrics.balanced_accuracy_score,
            cv_method=model_selection.KFold(n_splits=3, shuffle=True),
            # cv_method="NonShuffledTrainTestSplit",
            get_movement_detection_rate=False,
            min_consequent_count=2,
            TRAIN_VAL_SPLIT=False,
            RUN_BAY_OPT=False,
            bay_opt_param_space=None,
            use_nested_cv=False,
            fs=feature_reader.settings["sampling_rate_features_hz"],
            oversampling=False,
            undersampling=True,
        )

        performances = feature_reader.run_ML_model(
            estimate_channels=True,
            estimate_gridpoints=False,
            estimate_all_channels_combined=True,
            save_results=True,
        )

        feature_reader.label = feature_reader.feature_arr.iloc[:, -1]
        feature_reader.label_name = feature_reader.feature_arr.columns[-1]

        class_labels = ["rest", "ntr", "pls", "unpls"]

        # Problem hier: all_ch
        ch_names = list(feature_reader.nm_channels.query("used == 1")["name"])
        ch_names.append("all_ch_combined")

        # plot feature importances
        def get_mi_scores(df, label):
            mi_res = mutual_info_classif(df, label)
            mi_series = pd.Series(mi_res, df.columns)
            mi_series_sorted = mi_series.sort_values(ascending=False)
            return mi_series_sorted

        # 4 is the columns ALL, so remove everything from df where ALL is not UNPLS
        df_f_unpls_rest = feature_reader.decoder.features.iloc[
            np.where(
                (feature_reader.decoder.features.iloc[:, -4] == 2)
                | (feature_reader.decoder.features.iloc[:, -4] == 3)
            )[0]
        ]
        mi_series_sorted = get_mi_scores(
            df_f_unpls_rest.iloc[:, :-5].fillna(0),
            df_f_unpls_rest.iloc[:, -1],
        )

        for ch_name in list(ch_names):
            if ch_name.startswith("all_ch") is False:
                df_filtered = feature_reader.feature_arr[
                    feature_reader.filter_features(
                        feature_reader.feature_arr.columns, ch_name, None
                    )
                ]
                data = np.expand_dims(np.array(df_filtered), axis=1)

                X_epoch, y_epoch = feature_reader.get_epochs(
                    data,
                    feature_reader.label,
                    epoch_len=4,
                    sfreq=feature_reader.settings["sampling_rate_features_hz"],
                    threshold=0.1,
                )
                y_test = np.concatenate(
                    feature_reader.decoder.ch_ind_results[ch_name]["y_test"]
                )
                y_test_pr = np.concatenate(
                    feature_reader.decoder.ch_ind_results[ch_name]["y_test_pr"]
                )
            else:
                X_epoch = None
                y_epoch = None
                y_test = np.concatenate(feature_reader.decoder.all_ch_results["y_test"])
                y_test_pr = np.concatenate(
                    feature_reader.decoder.all_ch_results["y_test_pr"]
                )
            cm = confusion_matrix(
                y_test,
                y_test_pr,
                normalize="true",
            )

            res_dict[RUN_NAME][ch_name] = {}
            res_dict[RUN_NAME][ch_name]["cm"] = cm
            res_dict[RUN_NAME][ch_name]["X_epoch"] = X_epoch
            res_dict[RUN_NAME][ch_name]["y_epoch"] = y_epoch
            res_dict[RUN_NAME][ch_name]["performances"] = performances[""][ch_name][
                "performance_test"
            ]
            res_dict[RUN_NAME][ch_name]["y_test"] = y_test
            res_dict[RUN_NAME][ch_name]["y_test_pr"] = y_test_pr
            res_dict[RUN_NAME][ch_name]["mi_series_sorted"] = mi_series_sorted

        PLT_ = False
        if PLT_:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=class_labels
            )
            disp.plot()

            vals_plt = 20
            plt.bar(np.arange(vals_plt), mi_series_sorted.values[:vals_plt])
            plt.xticks(
                np.arange(vals_plt), mi_series_sorted.index[:vals_plt], rotation=90
            )
            plt.tight_layout()
            plt.show()

            # check for NaN's
            plt.imshow(feature_reader.decoder.features, aspect="auto")
            plt.clim(-1, 1)
            # without NaN's
            df_without_nan = feature_reader.decoder.features[
                feature_reader.decoder.features["Cg25R01_fooof_a_exp"].notna()
            ].copy()
            mi_series_sorted_without_nan = get_mi_scores(
                df_without_nan.iloc[:, :23], df_without_nan.iloc[:, -4]
            )

        print(f"Analysis finished {ch_name}")

    print("start plotting here")
    with open("dict_res_out_PLS_UNPLS_MI.p", "wb") as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
