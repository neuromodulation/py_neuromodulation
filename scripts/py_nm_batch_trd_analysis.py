from py_neuromodulation import nm_analysis, nm_decode
import numpy as np
from sklearn import metrics, linear_model, model_selection


def main():
    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\features"

    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file="effspm8_JUN_EMO"
    )

    PLT = False
    if PLT:

        feature_used = "fft"

        feature_reader.label = feature_reader.feature_arr["label_UNPLS"]
        feature_reader.label = np.array(feature_reader.feature_arr["label_UNPLS"])

        ch_used = feature_reader.nm_channels.query(
            '(type=="lfp") and (used == 1)'
        ).iloc[0]["name"]

        feature_reader.plot_target_averaged_channel(
            ch=ch_used, list_feature_keywords=[feature_used], epoch_len=7, threshold=0.5
        )

    feature_reader.label_name = "label"
    feature_reader.label = np.array(feature_reader.feature_arr["label"])

    model = linear_model.LogisticRegression()

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
        fs=feature_reader.settings["sampling_rate_features"],
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=False,
        estimate_all_channels_combined=True,
        save_results=True,
    )


if __name__ == "__main__":
    main()
