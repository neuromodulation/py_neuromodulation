import os
import pandas as pd
from sklearn import metrics, linear_model, model_selection
from py_neuromodulation import nm_analysis, nm_decode, nm_plots


def analyze_nolds():
    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\test_nolds"
    RUN_NAME = "sub-000_ses-right_task-force_run-0_ieeg"

    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=RUN_NAME
    )

    ch_used = feature_reader.nm_channels.query(
        '(type=="ecog") and (used == 1)'
    ).iloc[0]["new_name"]

    for features_plt in [["beta"], ["gamma"], ["fft"], None]:

        feature_reader.plot_target_averaged_channel(
            ch=ch_used,
            list_feature_keywords=features_plt,
            epoch_len=4,
            threshold=0.5,
        )

    # plot them individually per every fband

    list_df_per = []
    features_fft = feature_reader.feature_arr[
        [f for f in feature_reader.feature_arr.columns if "fft" in f]
    ]
    features_nolds = feature_reader.feature_arr[
        [
            f
            for f in feature_reader.feature_arr.columns
            if "fft" not in f and "MOV" not in f and "time" not in f
        ]
    ]
    features_nolds_raw = feature_reader.feature_arr[
        [f for f in feature_reader.feature_arr.columns if "raw" in f]
    ]
    features_all = feature_reader.feature_arr
    for features_, feature_names in [
        (features_fft, "fft"),
        (features_nolds, "nolds"),
        (features_nolds_raw, "nolds_raw"),
        (features_all, "all"),
    ]:
        feature_reader.decoder = nm_decode.Decoder(
            features=features_,
            label=feature_reader.label,
            label_name=feature_reader.label_name,
            used_chs=feature_reader.used_chs,
            model=linear_model.LogisticRegression(),
            eval_method=metrics.balanced_accuracy_score,
            cv_method=model_selection.KFold(n_splits=3, shuffle=True),
            get_movement_detection_rate=True,
            min_consequent_count=2,
            TRAIN_VAL_SPLIT=False,
            RUN_BAY_OPT=False,
            bay_opt_param_space=None,
            use_nested_cv=True,
            fs=feature_reader.settings["sampling_rate_features_hz"],
        )
        performances = feature_reader.run_ML_model(
            estimate_channels=True,
            estimate_gridpoints=False,
            estimate_all_channels_combined=True,
            save_results=True,
        )
        df_per = feature_reader.get_dataframe_performances(performances)
        df_per["features"] = feature_names
        list_df_per.append(df_per)

    df = pd.concat(list_df_per)
    nm_plots.plot_df_subjects(
        df,
        x_col="ch",
        y_col="performance_test",
        hue="features",
        PATH_SAVE=os.path.join(PATH_OUT, RUN_NAME, "plt_df.png"),
    )


if __name__ == "__main__":
    analyze_nolds()
