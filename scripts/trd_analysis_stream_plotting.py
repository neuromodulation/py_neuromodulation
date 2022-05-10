from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import os
import sys

from py_neuromodulation import nm_plots, nm_analysis

sys.path.append(r"C:\Users\ICN_admin\Documents\icn\icn_stats")
import icn_stats


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


def main():

    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\features_epochs_nonorm"
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file="effspm8_JUN_EMO", binarize_label=False
    )

    with open("dict_res_out_PLS_UNPLS_MI.p", "rb") as handle:
        d = pickle.load(handle)

    class_labels = ["rest", "ntr", "pls", "unpls"]

    depression_scores = {
        "BDI24_am": {
            "JUN": 37,
            "KOR": 36,
            "MIC": 32,
            "NIL": 37,
            "OHL": 1,
            "SCH": 31,
            "THI": 12,
            "WES": 52,
        },
        "HAMD24_24": {
            "JUN": 33,
            "KOR": 27,
            "MIC": 22,
            "NIL": 29,
            "OHL": 9,
            "SCH": 16,
            "THI": 7,
            "WES": 33,
        },
    }

    df = pd.DataFrame()
    for run in d.keys():
        for ch in d[run].keys():

            if ch.startswith("all_ch_"):
                ALL_COMB = True
            else:
                ALL_COMB = False
            df = df.append(
                {
                    "performance_test": d[run][ch]["performances"],
                    "sub": run[len("effspm8_") : len("effspm8_") + 3],
                    "ch": ch,
                    "all_comb": ALL_COMB,
                    "BDI": depression_scores["BDI24_am"][
                        run[len("effspm8_") : len("effspm8_") + 3]
                    ],
                    "HAMD": depression_scores["HAMD24_24"][
                        run[len("effspm8_") : len("effspm8_") + 3]
                    ],
                },
                ignore_index=True,
            )

    print("plot mean performances")

    # plot depression correlation
    metric = "BDI"
    df_plt = df.query("all_comb == False").groupby("sub").apply("mean")
    df_plt = df.query("all_comb == True")
    rho, p = icn_stats.permutationTestSpearmansRho(
        df_plt["performance_test"],
        df_plt[metric],
        False,
        "R^2",
        5000,
    )
    sb.regplot(x="performance_test", y=metric, data=df_plt)
    plt.title(f"Accuracy~{metric} p={np.round(p, 2)} rho={np.round(rho, 2)}")

    cm = []
    x_epoch = []
    y_epoch = []
    y_pr = []
    y_te = []

    mi_series = []

    for run in d.keys():
        for ch in d[run].keys():
            cm.append(d[run][ch]["cm"])
            if ch.startswith("all") is False:
                x_epoch.append(np.squeeze(d[run][ch]["X_epoch"].mean(axis=0)))
                y_epoch.append(d[run][ch]["y_epoch"].mean(axis=0))
            else:
                y_pr.append(d[run][ch]["y_test_pr"])
                y_te.append(d[run][ch]["y_test"])
        mi_series.append(d[run][ch]["mi_series_sorted"])

    # time plot
    y_pr_ = np.concatenate(y_pr)
    y_te_ = np.concatenate(y_te)
    y_te_[np.where(y_te_ != 3)[0]] = 0
    X_, y_ = feature_reader.get_epochs(
        np.expand_dims(y_pr_, axis=(1, 2)), y_te_, epoch_len=4, sfreq=10, threshold=0.1
    )
    y_unpls = np.squeeze(X_).mean(axis=0)

    # 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
    y_te_ = np.concatenate(y_te)
    y_te_[np.where(y_te_ != 2)[0]] = 0
    X_, y_ = feature_reader.get_epochs(
        np.expand_dims(y_pr_, axis=(1, 2)), y_te_, epoch_len=4, sfreq=10, threshold=0.1
    )
    y_pls = np.squeeze(X_).mean(axis=0)

    y_te_ = np.concatenate(y_te)
    y_te_[np.where(y_te_ != 1)[0]] = 0
    X_, y_ = feature_reader.get_epochs(
        np.expand_dims(y_pr_, axis=(1, 2)), y_te_, epoch_len=4, sfreq=10, threshold=0.1
    )
    y_ntr = np.squeeze(X_).mean(axis=0)

    y_te_ = np.concatenate(y_te)
    y_te_[np.where(y_te_ == 0)[0]] = 5
    y_te_[np.where(y_te_ != 5)[0]] = 0
    X_, y_ = feature_reader.get_epochs(
        np.expand_dims(y_pr_, axis=(1, 2)), y_te_, epoch_len=4, sfreq=10, threshold=0.1
    )
    y_rest = np.squeeze(X_).mean(axis=0)

    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(y_rest, linewidth=2, color="black", label="rest")
    plt.plot(y_ntr, linewidth=2, color="gray", label="ntr")
    plt.plot(y_pls, linewidth=2, color="blue", label="pls")
    plt.plot(y_unpls, linewidth=2, color="green", label="unpls")
    plt.xticks(np.arange(0, 40, 2), np.round(np.arange(-2, 2, 0.2), 2))
    plt.xlabel("Time [s]")
    plt.legend()
    plt.ylabel("Prediction")
    plt.title(
        "XGBOOST mean predictions all channels\n 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS"
    )
    plt.tight_layout()
    plt.show()

    print("plot")

    # plot best
    best_series = d["effspm8_SCH_EMO"]["Cg25L23"]["mi_series_sorted"]
    best_series = best_series.reset_index()
    best_series[best_series["index"].str.startswith("Cg25L23")].set_index(
        "index"
    ).plot.bar()
    plt.ylabel("Mutual Information")
    plt.title("Sorted Best subject ch Cg25L23 MI feature scores")
    plt.tight_layout()

    # plot mean all
    df_mean_sorted = (
        pd.concat(mi_series, axis=1).mean(axis=1).sort_values(ascending=False)
    )
    df_mean_sorted.iloc[:20].plot.bar()
    plt.ylabel("Mutual Information")
    plt.title("Sorted Mean subject MI feature scores")
    plt.tight_layout()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm).mean(axis=0), display_labels=class_labels
    )
    disp.plot()

    # plot best confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(d["effspm8_SCH_EMO"]["Cg25L23"]["cm"]),
        display_labels=class_labels,
    )
    disp.plot()

    plot_df_subjects(
        df, PATH_OUT, x_col="sub", y_col="performance_test", hue="all_comb"
    )

    # have a look at the mi rates scors: in total there are 23 features per channel
    # take average series of all subjects and sort?

    # plot best
    x_best = stats.zscore(
        np.nan_to_num(
            np.squeeze(d["effspm8_SCH_EMO"]["Cg25L23"]["X_epoch"].mean(axis=0))
        ),
        axis=1,
    )
    y_best = d["effspm8_SCH_EMO"]["Cg25L23"]["y_epoch"]
    plot_epoch(x_best.T, y_best, x_best.T)

    # plot all
    plot_epoch(x_epoch, y_epoch)

    def plot_epoch(x_epoch, y_epoch, X_epoch_mean=None):
        epoch_len = 4
        feature_names = [f[8:] for f in feature_reader.feature_arr.columns[:23]]
        label_name = "UNPLS"
        X_epoch = np.array(x_epoch)
        sfreq = 10
        if X_epoch_mean is None:
            X_epoch_mean = stats.zscore(
                np.nan_to_num(np.nanmean(np.squeeze(X_epoch), axis=0)), axis=0
            ).T
        y_epoch = np.stack(np.array(y_epoch))
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.imshow(X_epoch_mean, aspect="auto")
        plt.yticks(np.arange(0, len(feature_names), 1), feature_names)
        plt.xticks(
            np.arange(0, X_epoch.shape[1], 1),
            np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2),
            rotation=90,
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Time [s]")
        str_title = "UNPLS aligned features mean all channels"
        plt.title(str_title)

        plt.subplot(212)
        for i in range(y_epoch.shape[0]):
            plt.plot(y_epoch[i, :], color="black", alpha=0.4)
        plt.plot(
            y_epoch.mean(axis=0),
            color="black",
            alpha=1,
            linewidth=3.0,
            label="mean target",
        )
        plt.legend()
        plt.ylabel("target")
        plt.title(label_name)
        plt.xticks(
            np.arange(0, X_epoch.shape[1], 1),
            np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2),
            rotation=90,
        )
        plt.xlabel("Time [s]")
        plt.tight_layout()


if __name__ == "__main__":
    main()
