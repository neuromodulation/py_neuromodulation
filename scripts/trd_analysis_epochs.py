import pickle
import os

import xgboost
from py_neuromodulation import nm_analysis, nm_decode
import numpy as np
import pandas as pd
from sklearn import metrics, linear_model, model_selection, svm
from matplotlib import pyplot as plt
from scipy import stats
from xgboost import XGBClassifier
import seaborn as sb


def plot_epoch_avg_features(
    ch_name: str,
    features: np.array,
    label: np.array,
    feature_names: list,
    PATH_OUT: str,
    file_name: str,
    stimuli_to_plot: list = ["PLS", "UNPLS", "NTR"],
):
    """_summary_

    Parameters
    ----------
    ch_name : str
        select a single channel for plotting
    features : np.array
        shape (number_epochs, time, channel)
    label : np.array
        label array
    feature_names : list
        feature names, same for each channel
    PATH_OUT : str
        _description_
    file_name : str
        _description_
    stimuli_to_plot : list, optional
        _description_, by default ["PLS", "UNPLS", "NTR"]
    """
    idx_ch = [i for i in range(len(feature_names)) if ch_name in feature_names[i]]
    feature_names_ch = np.array(feature_names)[idx_ch]

    plt.figure(figsize=(13, 4), dpi=300)

    for stim_idx, stim in enumerate(stimuli_to_plot):

        data_ch_pls = features[np.array(label) == stim, :, :][:, :, idx_ch]

        plt.subplot(1, 3, stim_idx + 1)
        plt.imshow(
            stats.zscore(data_ch_pls.mean(axis=0), axis=0).T,
            aspect="auto",
        )
        if stim_idx == 0:
            plt.yticks(range(len(idx_ch)), feature_names_ch)
        else:
            plt.yticks(color="w")
        plt.xticks(
            np.arange(0, data_ch_pls.shape[1], 5),
            np.arange(-2.5, 3.6, 0.5),
        )
        plt.xlabel("Time [s]")
        plt.xlim(0, 50)  # to cut off the last 1s due to preprocessing

        plt.gca().invert_yaxis()
        plt.title(f"{ch_name} {stim}")
        plt.savefig(
            os.path.join(PATH_OUT, file_name, f"{ch_name}_avg_features.png"),
            bbox_inches="tight",
        )
        plt.close()


def plot_df_subjects(
    df, PATH_OUT, x_col="sub", y_col="performance_test", hue="all combined"
):
    alpha_box = 0.4
    plt.figure(figsize=(5, 3), dpi=300)
    sb.boxplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        showmeans=False,
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        whiskerprops={"linewidth": 2, "zorder": 10, "alpha": alpha_box},
        capprops={"alpha": alpha_box},
        medianprops=dict(linestyle="-.", linewidth=5, color="gray", alpha=alpha_box),
    )

    ax = sb.stripplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        dodge=True,
        s=5,
    )

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(
        handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )
    plt.title("subject specific channel performances")
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    plt.savefig(
        os.path.join(PATH_OUT, "all_sub_performances.png"),
        bbox_inches="tight",
    )


def plot_bar_performance_per_channel(
    ch_names, performances: dict, PATH_OUT: str, file_name: str
):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.bar(
        np.arange(len(ch_names)),
        [performances[""][p]["performance_test"] for p in performances[""]],
    )
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.xlabel("channels")
    plt.ylabel("Balanced Accuracy")
    plt.savefig(
        os.path.join(PATH_OUT, file_name, "XGB_performance_comparison_ch.png"),
        bbox_inches="tight",
    )
    plt.close()


def main():
    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\features"
    files = [f for f in os.listdir(PATH_OUT) if f.startswith("effspm8")]
    performances_all = {}
    df_plt = pd.DataFrame()

    for file_name in files:
        # file_name = "effspm8_JUN_EMO"
        if "KSC" in file_name:
            continue
        try:
            feature_reader = nm_analysis.Feature_Reader(
                feature_dir=PATH_OUT, feature_file=file_name
            )

            # read pickle file here with label and data across epoch
            with open(os.path.join(PATH_OUT, file_name, "dict_out.p"), "rb") as fp:
                dict_out = pickle.load(fp)

            PLT_ = False
            if PLT_ is True:
                # dict_out["data"] with shape (trials, time, num_features)
                for ch_name in [
                    ch for ch in feature_reader.nm_channels["name"] if "label" not in ch
                ]:

                    plot_epoch_avg_features(
                        ch_name,
                        dict_out["features"],
                        dict_out["label"],
                        dict_out["feature_names"],
                        PATH_OUT,
                        file_name,
                        stimuli_to_plot=["PLS", "UNPLS", "NTR"],
                    )

            # construct label out of epochs

            feature_ch_dfs = []
            ch_names = [
                ch for ch in feature_reader.nm_channels["name"] if "label" not in ch
            ]
            for ch_name in ch_names:
                idx_ch = [
                    i
                    for i in range(len(dict_out["feature_names"]))
                    if ch_name in dict_out["feature_names"][i]
                ]
                feature_names_ch = np.array(dict_out["feature_names"])[idx_ch]

                # select time interval for positive label class
                # and reshape to continuous stream
                feature_arr = dict_out["features"][:, np.arange(25, 35, 1), :][
                    :, :, idx_ch
                ].reshape(dict_out["features"].shape[0], int(10 * len(idx_ch)))

                features_after_stim = dict_out["features"][:, np.arange(25, 35, 1), :][
                    :, :, idx_ch
                ]

                feature_arr = features_after_stim.reshape(
                    int(features_after_stim.shape[0] * 10), len(idx_ch)
                )

                feature_cols = []
                for feature_name_ch in range(len(feature_names_ch)):
                    for time in range(10):
                        feature_cols.append(
                            f"{feature_names_ch[feature_name_ch]}_time_{time}"
                        )
                # feature_cols = []
                # for feature_name_ch in feature_names_ch:
                #    feature_cols.append(feature_name_ch[feature_name_ch.find('_')+1:])

                feature_ch_dfs.append(
                    pd.DataFrame(data=feature_arr, columns=feature_names_ch)
                )

            # model = linear_model.LogisticRegression()
            model = svm.SVC(kernel="linear")
            # model = xgboost.XGBClassifier()

            decoder_feature_arr = pd.concat(feature_ch_dfs, axis=1)
            decoder_label = np.repeat(dict_out["label"], 10)

            KEEP_NTR = False
            if KEEP_NTR == False:
                decoder_feature_arr = decoder_feature_arr.iloc[
                    np.where(decoder_label != "NTR")[0], :
                ]
                decoder_label = decoder_label[np.where(decoder_label != "NTR")[0]]
                decoder_label[np.where(decoder_label == "PLS")] = 0
                decoder_label[np.where(decoder_label == "UNPLS")] = 1
                decoder_label = np.array(decoder_label, dtype="bool")

            feature_reader.decoder = nm_decode.Decoder(
                features=decoder_feature_arr,
                label=decoder_label,
                label_name="label",
                used_chs=ch_names,
                model=model,
                eval_method=metrics.balanced_accuracy_score,
                cv_method=model_selection.KFold(n_splits=3, shuffle=True),
                # cv_method="NonShuffledTrainTestSplit",
                get_movement_detection_rate=True,
                min_consequent_count=3,
                TRAIN_VAL_SPLIT=False,
                RUN_BAY_OPT=False,
                STACK_FEATURES_N_SAMPLES=False,
                bay_opt_param_space=None,
                save_coef=True,
                use_nested_cv=False,
                fs=feature_reader.settings["sampling_rate_features_hz"],
            )

            performances = feature_reader.run_ML_model(
                estimate_channels=True,
                estimate_gridpoints=False,
                estimate_all_channels_combined=True,
                save_results=True,
            )
            if PLT_ is True:
                plot_bar_performance_per_channel(
                    ch_names, performances, PATH_OUT, file_name
                )

            for ch_name in performances[""].keys():
                ALL_COMB = False
                if ch_name.startswith("all"):
                    ALL_COMB = True
                dict_add = performances[""][ch_name]
                dict_add["sub"] = file_name[8:-4]
                dict_add["ch"] = ch_name
                dict_add["all combined"] = ALL_COMB
                dict_add["feature_names"] = [
                    f[feature_names_ch[0].find("_") + 1 :] for f in feature_names_ch
                ]
                df_plt = df_plt.append(
                    dict_add,
                    ignore_index=True,
                )

            performances_all[file_name] = [
                performances[""][p]["performance_test"] for p in performances[""]
            ]

        except:
            print(f"error at {file_name}")

    plot_df_subjects(df_plt, PATH_OUT, x_col="sub", y_col="mov_detection_rates_test")
    plot_df_subjects(df_plt, PATH_OUT, x_col="sub", y_col="performance_test")

    # get the SVM coefficients of maximum performance_test
    df_ch = df_plt[df_plt["all combined"] == 0]
    idx_max = (
        df_ch.groupby("sub")["performance_test"].transform(max)
        == df_ch["performance_test"]
    )
    df_max_idx = df_ch[idx_max]
    coef_ = np.vstack(df_max_idx["coef"])

    df_plt_coef = pd.DataFrame(data=coef_, columns=df_plt.iloc[0]["feature_names"])
    df_plt_coef = df_plt_coef.stack().reset_index()
    df_plt_coef = df_plt_coef.rename(
        columns={"level_1": "feature_name", 0: "svm_coefficient_value"}
    )
    plot_df_subjects(
        df_plt_coef,
        PATH_OUT,
        x_col="feature_name",
        y_col="svm_coefficient_value",
        hue=None,
    )

    plt.boxplot(coef_)
    plt.xticks(np.arange(9), df_plt.iloc[0]["feature_names"])

    all_per = np.concatenate([performances_all[p] for p in performances_all.keys()])
    plt.hist(all_per)
    plt.xlabel("Balanced Accuracy")
    plt.title("All channel performances")


if __name__ == "__main__":
    main()
