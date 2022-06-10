import enum
from py_neuromodulation import nm_decode, nm_analysis, nm_plots

import pandas as pd
import numpy as np
import os
import _pickle as cPickle
import xgboost
import catboost
import seaborn as sb
from sklearn import (
    metrics,
    linear_model,
    model_selection,
    ensemble,
    preprocessing,
)
from matplotlib import pyplot as plt
from imblearn import over_sampling


def plot_LM_coeff():

    mean_coef_lm = np.stack(
        pd.concat(pd_data).query("all_combined == 0")["coef"]
    ).mean(axis=0)
    feature_names = [
        f[8:] for f in analyzer.decoder.feature_names if f.startswith("Cg25R01")
    ]

    plt.figure(figsize=(15, 10), dpi=300)
    sort_idx = np.argsort(mean_coef_lm)[::-1]
    plt.bar(np.arange(len(feature_names)), mean_coef_lm[sort_idx])
    plt.xticks(
        np.arange(len(feature_names)),
        np.array(feature_names)[sort_idx],
        rotation=90,
    )
    plt.ylabel("Linear Model Mean coefficients")
    plt.title("Mean channel and subject LM coefficients")
    plt.tight_layout()
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\LM_coeff_single_ch_mean.png"
    )


def plot_performances_across_methods():
    df_all = pd.read_csv(
        "all_fatures_lm.csv"
    )  # csv unfortunately does not save the coef array correctly
    df_pca = pd.read_csv("all_fatures_pca.csv")
    df_pca_xgb = pd.read_csv("all_fatures_pca_xgb.csv")
    df_cca = pd.read_csv("all_fatures_cca.csv")
    df_cca_xgb = pd.read_csv("all_fatures_cca_xgb.csv")
    df_all_xgb = pd.read_csv("all_fatures_xgb.csv")
    df_mrmr = pd.read_csv("mrmr_fatures.csv")
    df_mrmr_20 = pd.read_csv("mrmr_fatures_20.csv")
    df_mrmr_xgb = pd.read_csv("mrmr_features_xgb.csv")
    df_fft = pd.read_csv("fft_fatures.csv")
    df_fft_xgb = pd.read_csv("fft_features_xgb.csv")

    df_all["features_used"] = "all_lm"
    df_pca["features_used"] = "pca_lm"
    df_pca_xgb["features_used"] = "pca_xgb"
    df_cca["features_used"] = "cca_lm"
    df_cca_xgb["features_used"] = "cca_xgb"
    df_all_xgb["features_used"] = "all_xgb"
    df_mrmr["features_used"] = "mrmr_10_lm"
    df_mrmr_20["features_used"] = "mrmr_20_lm"
    df_mrmr_xgb["features_used"] = "mrmr_10_xgb"
    df_fft["features_used"] = "fft_lm"
    df_fft_xgb["features_used"] = "fft_xgb"

    df_comb = pd.concat(
        [
            df_fft,
            df_fft_xgb,
            df_all,
            df_all_xgb,
            df_pca,
            df_pca_xgb,
            df_cca,
            df_cca_xgb,
            df_mrmr,
            df_mrmr_20,
            df_mrmr_xgb,
        ]
    )
    df_mean_sub_ind_ch = (
        df_comb.query("all_combined == 0")
        .groupby(["features_used", "sub"])
        .mean()
        .reset_index((0, 1))
    )
    df_plt = pd.concat([df_comb.query("all_combined == 1"), df_mean_sub_ind_ch])

    nm_plots.plot_df_subjects(
        df=df_plt,
        y_col="performance_test",
        x_col="all_combined",
        hue="features_used",
        PATH_SAVE=r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\LM_featuresused_comparison_mrmr_all_fft.png",
    )


def plot_channels_selected_mrmr():
    df_use = pd.concat(pd_data).query("all_combined == 0")
    used_features = np.squeeze(np.stack(df_use["mrmr_select"])).flatten()
    val_cnt = pd.Series([f[8:] for f in used_features]).value_counts()
    plt.figure(figsize=(10, 10), dpi=300)
    val_cnt.plot.pie()
    plt.title("#times selected features")
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\MRMR_FeaturesSelected.png"
    )


def fix_name_columns(feature_name: str):
    if feature_name.startswith("burst"):
        feature_str = "bursts"
    elif feature_name.startswith("nolds"):
        feature_str = "nolds"
    else:
        return feature_name
    str_start = feature_name.find("_") + 1
    str_end = feature_name.find("_") + 8
    ch_name = feature_name[str_start:str_end]
    feature_name_new = ch_name + "_" + feature_str + feature_name[str_end:]
    return feature_name_new


def plot_correlation_BDI():
    df = pd.read_csv("mrmr_fatures.csv")  # mrmr_fatures
    df_comp = df.query("all_combined == 1")[
        ["performance_test", "sub"]
    ].reset_index()
    df_comp["BDI"] = [33, 46, 35, 41, 22, 36, 57]
    df_comp["BDI_24"] = [35, 36, 32, 37, 1, 12, 52]
    df_comp["BDI_DIFF"] = df_comp["BDI"] - df_comp["BDI_24"]
    nm_plots.reg_plot(x_col="BDI_DIFF", y_col="performance_test", data=df_comp)
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\DIFF_BDI_per_plt.png"
    )


PATH_FEATURES = r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05"

subjects = [
    f
    for f in os.listdir(PATH_FEATURES)
    if f.startswith("effspm8") and "KSC" not in f
]

pd_data = []

PLS = -2
UNPLS = -1
NTR = -3
ALL = -4

LABEL = ALL
mean_acc = []
epoch_out = {}
data_all = []
label_all = []

for sub in subjects:
    analyzer = nm_analysis.Feature_Reader(
        feature_dir=PATH_FEATURES,
        feature_file=sub,
        binarize_label=False,
    )

    analyzer.label = np.array(
        analyzer.feature_arr.loc[analyzer.feature_arr["ALL"] != 0, :].iloc[
            :, LABEL
        ]
    )

    analyzer.feature_arr = analyzer.feature_arr.loc[
        analyzer.feature_arr["ALL"] != 0, :
    ].iloc[:, :-5]

    data_all.append(analyzer.feature_arr)
    label_all.append(analyzer.label)
    # analyzer.label = analyzer.feature_arr[
    #    "label"
    # ]  # preprocessing.LabelEncoder().fit_transform(analyzer.feature_arr["label"])
    # analyzer.label_name = "label"

    # feature_names = [f for f in analyzer.feature_arr.columns if "fft" in f]

    # analyzer.feature_arr = analyzer.feature_arr[feature_names]

    analyzer.feature_arr.columns = list(
        map(fix_name_columns, analyzer.feature_arr.columns)
    )

    analyzer.set_decoder(
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        save_coef=True,
        model=linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced"
        ),  #
        # model=xgboost.XGBClassifier(),  # ,   # catboost.CatBoostClassifier(),
        eval_method=metrics.balanced_accuracy_score,
        cv_method="NonShuffledTrainTestSplit",  # model_selection.KFold(
        # n_splits=3, random_state=None, shuffle=False
        # ),
        get_movement_detection_rate=False,
        min_consequent_count=3,
        threshold_score=False,
        bay_opt_param_space=None,
        STACK_FEATURES_N_SAMPLES=False,
        time_stack_n_samples=5,
        use_nested_cv=False,
        VERBOSE=False,
        undersampling=False,
        oversampling=True,
        mrmr_select=False,
        cca=False,
        pca=False,
    )

    analyzer.decoder.feature_names = list(analyzer.decoder.features.columns)

    performances = analyzer.run_ML_model(
        estimate_channels=True, estimate_all_channels_combined=True
    )

    df = analyzer.get_dataframe_performances(performances)
    df["sub"] = sub
    pd_data.append(df)

    # reset all unpleasant ones
    # integer_encoded_names: 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
    all_sub_label_epochs = {}
    for label_ in [1, 2, 3]:
        y_te = analyzer.label.copy()
        y_te[np.where(y_te != label_)[0]] = 0
        y_te[np.where(y_te == label_)[0]] = 1
        X_feature_epochs_all_ch, _ = analyzer.get_epochs(
            np.expand_dims(np.array(analyzer.feature_arr), axis=1),
            np.array(y_te),
            epoch_len=2,
            sfreq=10,
            threshold=0.1,
        )
        f_epochs = np.squeeze(X_feature_epochs_all_ch)
        # get the channl specific epochs and add them to the dict

        for ch in [
            "Cg25R01",
            "Cg25R12",
            "Cg25R23",
            "Cg25R03",
            "Cg25L01",
            "Cg25L12",
            "Cg25L23",
            "Cg25L03",
        ]:
            ch_columns = [
                idx
                for idx, c in enumerate(analyzer.feature_arr.columns)
                if c.startswith(ch)
            ]
            ch_feature_names = [
                c[len(ch) + 1 :]
                for idx, c in enumerate(analyzer.feature_arr.columns)
                if c.startswith(ch)
            ]
            if label_ not in epoch_out:
                epoch_out[label_] = f_epochs[:, :, ch_columns]
            else:
                epoch_out[label_] = np.concatenate(
                    (epoch_out[label_], f_epochs[:, :, ch_columns]), axis=0
                )

    label_out = {}
    for label_ in [1, 2, 3]:
        y_te = np.concatenate(analyzer.decoder.all_ch_results["y_test"])
        y_te_pr = np.concatenate(analyzer.decoder.all_ch_results["y_test_pr"])
        y_te[np.where(y_te != label_)[0]] = 0
        y_te[np.where(y_te == label_)[0]] = 1
        y_te_pr_epochs, y_te_epochs = analyzer.get_epochs(
            np.expand_dims(y_te_pr, axis=(1, 2)),
            y_te,
            epoch_len=6,
            sfreq=10,
            threshold=0.1,
        )
        y_te_pr_epochs = np.squeeze(y_te_pr_epochs)
        acc_ = (
            np.sum(y_te_pr_epochs[:, 30:] == label_, axis=0)
            / y_te_epochs.shape[0]
        )
        label_out[label_] = acc_
        label_out["label"] = y_te_epochs.mean(axis=0)

    mean_acc.append(label_out)


# get count of best features
# pd.Series(np.concatenate(np.concatenate(np.stack(pd.concat(pd_data)["mrmr_select"])))).value_counts()

# 1 - NTR, 2 - PLS, 3 - UNPLS
for stim_idx, stim in ((1, "NEUTRAL"), (2, "PLEASANT"), (3, "UNPLEASANT")):
    plt.figure(figsize=(10, 10), dpi=300)
    out_plt = epoch_out[stim_idx].mean(axis=0) - epoch_out[stim_idx].mean(
        axis=0
    )[:10, :].mean(axis=0)
    plt.imshow(out_plt[10:, :].T, aspect="auto")
    plt.xticks(np.arange(10), np.round(np.arange(0, 1, 0.1), 2))
    plt.xlabel("Time [s]")
    plt.yticks(np.arange(out_plt.shape[1]), ch_feature_names)
    plt.title(f"{stim} AVG Features")
    plt.tight_layout()
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\MeanFeatures"
        + stim
        + ".png"
    )

analyzer.plot_target_averaged_channel(ch_names_ECOG="Cg25R01")

# plot mean accuracies
plt.subplot(121)
plt.plot(
    np.arange(0, 1, 0.1),
    np.stack([f[1] for f in mean_acc]).mean(axis=0)[:10],
    label="NTR",
)
plt.plot(
    np.arange(0, 1, 0.1),
    np.stack([f[2] for f in mean_acc]).mean(axis=0)[:10],
    label="PLS",
)
plt.plot(
    np.arange(0, 1, 0.1),
    np.stack([f[3] for f in mean_acc]).mean(axis=0)[:10],
    label="UNPLS",
)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Accuracy")
plt.title("predictions")

plt.subplot(122)
plt.plot(
    np.arange(-3, 3, 0.1), np.stack([f["label"] for f in mean_acc]).mean(axis=0)
)
plt.title("label")
plt.xlabel("Time [s]")

plt.show()


nm_plots.plot_df_subjects(
    df=pd.concat(pd_data),
    y_col="performance_test",
    x_col="sub",
    hue="all_combined",
    PATH_SAVE=r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\result_figures\LM_non_shuffled_mrmr.png",
)
plt.savefig("XGB_All_multi_class_ba.png", bbox_inches="tight")

plt.imshow(analyzer.feature_arr.T, aspect="auto")
plt.xticks(
    np.arange(0, analyzer.feature_arr.shape[0], 100),
    int(np.arange(0, analyzer.feature_arr.shape[0] / 10), 10),
)
plt.xlabel("Time [s]")
plt.ylabel("#Fetures")
plt.clim(-3, 3)

plt.tight_layout()

print("hallo")
