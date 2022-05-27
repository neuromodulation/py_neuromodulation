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

PATH_FEATURES = r"C:\Users\ICN_admin\Documents\TRD Analysis\features_epochs_realtime_norm_4classes"

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

    # analyzer.label = analyzer.feature_arr[
    #    "label"
    # ]  # preprocessing.LabelEncoder().fit_transform(analyzer.feature_arr["label"])
    # analyzer.label_name = "label"

    analyzer.set_decoder(
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        save_coef=False,
        model=linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced"
        ),  #
        # model=xgboost.XGBClassifier(),  # ,   # catboost.CatBoostClassifier(),
        eval_method=metrics.balanced_accuracy_score,
        cv_method=model_selection.KFold(
            n_splits=3, random_state=None, shuffle=False
        ),
        get_movement_detection_rate=True,
        min_consequent_count=3,
        threshold_score=False,
        bay_opt_param_space=None,
        STACK_FEATURES_N_SAMPLES=False,
        time_stack_n_samples=5,
        use_nested_cv=False,
        VERBOSE=False,
        undersampling=False,
        oversampling=True,
    )

    performances = analyzer.run_ML_model(
        estimate_channels=True, estimate_all_channels_combined=True
    )

    df = analyzer.get_dataframe_performances(performances)
    df["sub"] = sub
    pd_data.append(df)

    # reset all unpleasant ones
    # integer_encoded_names: 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
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


# plot mean accuracies
plt.subplot(121)
plt.plot(
    np.arange(0, 3, 0.1),
    np.stack([f[1] for f in mean_acc]).mean(axis=0),
    label="NTR",
)
plt.plot(
    np.arange(0, 3, 0.1),
    np.stack([f[2] for f in mean_acc]).mean(axis=0),
    label="PLS",
)
plt.plot(
    np.arange(0, 3, 0.1),
    np.stack([f[3] for f in mean_acc]).mean(axis=0),
    label="UNPLS",
)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Accuracy")
plt.title("predictions")

plt.subplot(122)
plt.plot(np.arange(-3, 3, 0.1), np.stack([f["label"] for f in mean_acc]).mean(axis=0))
plt.title("label")
plt.xlabel("Time [s]")

plt.show()


nm_plots.plot_df_subjects(
    df=pd.concat(pd_data),
    y_col="performance_test",
    x_col="sub",
    hue="all_combined",
)
plt.savefig("XGB_All_multi_class_ba.png", bbox_inches="tight")

plt.tight_layout()

print("hallo")
