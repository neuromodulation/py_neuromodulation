import os
from sklearn import linear_model, metrics, model_selection
from skopt import space as skopt_space
import xgboost
from py_neuromodulation import nm_BidsStream, nm_analysis


def run_example_BIDS():
    """run the example BIDS path in pyneuromodulation/tests/data
    """

    RUN_NAME = "sub-testsub_ses-EphysMedOff_task-buttonpress_run-0_ieeg.vhdr"
    PATH_RUN = os.path.join(
        os.path.abspath(os.path.join('examples','data')), 'sub-testsub', 'ses-EphysMedOff', 'ieeg',
        RUN_NAME)
    PATH_BIDS = os.path.abspath(os.path.join('examples','data'))
    PATH_OUT = os.path.abspath(os.path.join('examples', 'data', 'derivatives'))

    # read default settings
    nm_BIDS = nm_BidsStream.BidsStream(PATH_RUN=PATH_RUN,
                                       PATH_BIDS=PATH_BIDS,
                                       PATH_OUT=PATH_OUT,
                                       LIMIT_DATA=False,
                                       VERBOSE=False)

    nm_BIDS.run_bids()

    # init analyzer
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT,
        feature_file=RUN_NAME
    )

    # plot cortical signal
    feature_reader.plot_cort_projection()

    # plot for a single channel 
    ch_used = feature_reader.nm_channels.query(
        '(type=="ecog") and (used == 1)'
    ).iloc[0]["name"]

    feature_used = "stft"  if feature_reader.settings["methods"]["stft"] else "fft"

    feature_reader.plot_target_averaged_channel(
        ch=ch_used,
        list_feature_keywords=[feature_used],
        epoch_len=4,
        threshold=0.5
    )

    #model = linear_model.LogisticRegression(class_weight='balanced')
    model = xgboost.XGBClassifier(use_label_encoder=False)

    bay_opt_param_space = [
        skopt_space.Integer(1, 100, name='max_depth'),
        skopt_space.Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        skopt_space.Real(10**0, 10**1, "uniform", name="gamma")
    ]

    feature_reader.set_decoder(
        model = model,
        eval_method=metrics.balanced_accuracy_score,
        #cv_method=model_selection.KFold(n_splits=3, shuffle=True),
        cv_method="NonShuffledTrainTestSplit",
        get_movement_detection_rate=True,
        min_consequent_count=2,
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        bay_opt_param_space=bay_opt_param_space,
        use_nested_cv=True
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=True,
        estimate_all_channels_combined=True,
        save_results=True
    )

    # run here Bay. Opt.

    #performance_dict = feature_reader.read_results(read_grid_points=True, read_channels=True,
    #                                               read_all_combined=False,
    #                                               read_mov_detection_rates=True)
    feature_reader.plot_subject_grid_ch_performance(performance_dict=performances, plt_grid=True)
    