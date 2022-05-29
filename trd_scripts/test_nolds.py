import os

import nm_plots

import py_neuromodulation as nm
import xgboost
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
)
from sklearn import metrics, model_selection
from skopt import space as skopt_space


def set_settings(settings: dict):
    for method in list(settings["features"].keys()):
        settings["features"][method] = False

    settings["preprocessing"]["re_referencing"] = True
    settings["preprocessing"]["raw_reseample"] = True
    settings["preprocessing"]["notch_filter"] = True
    settings["preprocessing"]["raw_normalization"] = False
    settings["preprocessing"]["preprocessing_order"] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]

    settings["postprocessing"]["feature_normalization"] = True
    settings["postprocessing"]["project_cortex"] = False
    settings["postprocessing"]["project_subcortex"] = False

    settings["features"]["nolds"] = True
    settings["features"]["fft"] = True

    settings["nolds_features"]["data"]["raw"] = True
    settings["nolds_features"]["data"]["frequency_bands"] = [
        "theta",
        "alpha",
        "low beta",
        "high beta",
        "low gamma",
        "high gamma",
        "HFA",
    ]

    return settings


def run_example_BIDS():

    RUN_NAME = "sub-000_ses-right_task-force_run-0_ieeg"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-0_ieeg.vhdr"
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh"
    PATH_OUT = r"C:\Users\ICN_admin\Documents\TRD Analysis\test_nolds"

    (raw, data, sfreq, line_noise, _, _,) = nm_IO.read_BIDS_data(
        PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg"
    )

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "dbs"),
        target_keywords=("SQUARED_ROTATION",),
    )

    nm_channels["used"] = 0
    nm_channels.at[0, "used"] = 1  # select only one channel
    nm_channels.at[9, "target"] = 1  # target is squared rotation
    nm_channels.at[8, "target"] = 0

    stream = nm.Stream(
        settings=None,
        nm_channels=nm_channels,
        path_grids=None,
        verbose=True,
    )

    stream.settings = set_settings(stream.settings)

    stream.init_stream(
        sfreq=sfreq,
        line_noise=line_noise,
        coord_list=None,
        coord_names=None,
    )

    stream.run(
        data=data,
        out_path_root=PATH_OUT,
        folder_name=RUN_NAME,
    )

    # init analyzer
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=RUN_NAME
    )
    Anylze = False
    if Anylze is True:

        # plot for a single channel
        ch_used = feature_reader.nm_channels.query(
            '(type=="ecog") and (used == 1)'
        ).iloc[0]["name"]

        feature_used = (
            "stft" if feature_reader.settings["methods"]["stft"] else "fft"
        )

        feature_reader.plot_target_averaged_channel(
            ch=ch_used,
            list_feature_keywords=[feature_used],
            epoch_len=4,
            threshold=0.5,
        )

        model = xgboost.XGBClassifier(use_label_encoder=False)

        feature_reader.decoder = nm_decode.Decoder(
            features=feature_reader.feature_arr,
            label=feature_reader.label,
            label_name=feature_reader.label_name,
            used_chs=feature_reader.used_chs,
            model=model,
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

        nm_plots.plot_df_subjects(
            df_per, x_col="sub", y_col="performance_test", hue="all_combined"
        )


if __name__ == "__main__":

    run_example_BIDS()
