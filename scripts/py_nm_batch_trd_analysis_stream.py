import os
from sklearn import linear_model, metrics, model_selection, discriminant_analysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skopt import space as skopt_space
import xgboost
from py_neuromodulation import nm_GenericStream, nm_analysis, nm_decode


def main():

    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\features_epochs_nonorm"
    RUN_NAME = "effspm8_JUN_EMO"
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
    )

    model = linear_model.LogisticRegression()
    model = xgboost.XGBClassifier()
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

    class_labels = ["rest", "ntr", "pls", "unpls"]
    cm = confusion_matrix(
        feature_reader.decoder.ch_ind_results["Cg25R01"]["y_test"][0],
        feature_reader.decoder.ch_ind_results["Cg25R01"]["y_test_pr"][0],
        normalize="true"
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()

    # feature_reader.decoder.ch_ind_results["Cg25R01"]["y_test_prfeature_reader"
    print("Analysis finished")


if __name__ == "__main__":
    main()
