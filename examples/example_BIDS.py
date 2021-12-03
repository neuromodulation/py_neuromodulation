import multiprocessing
import sys
from bids import BIDSLayout
from itertools import product
import os
import json
from pathlib import Path
from sklearn import linear_model
from pyneuromodulation import nm_BidsStream, nm_analysis


def run_example_BIDS():
    """run the example BIDS path in pyneuromodulation/tests/data
    """

    PATH_RUN = os.path.join(
        os.path.abspath('examples\\data'), 'sub-testsub', 'ses-EphysMedOff', 'ieeg',
        "sub-testsub_ses-EphysMedOff_task-buttonpress_run-0_ieeg.vhdr")

    # read default settings
    nm_BIDS = nm_BidsStream.BidsStream(PATH_RUN=PATH_RUN,
                                       PATH_BIDS=os.path.abspath('examples\\data'),
                                       PATH_OUT=os.path.abspath(os.path.join('examples', 'data', 'derivatives')))

    nm_BIDS.run_bids()

    # plot features for ECoG channels
    feature_reader = nm_analysis.FeatureReadWrapper(feature_path=nm_BIDS.settings_wrapper.settings['out_path'],
                                                    plt_cort_projection=True)
    feature_reader.plot_features()
    model = linear_model.LogisticRegression(class_weight='balanced')
    feature_reader.run_ML_model(model=model, estimate_all_channels_combined=True)

    performance_dict = feature_reader.read_results(read_grid_points=True, read_channels=True,
                                                   read_all_combined=False,
                                                   read_mov_detection_rates=True)
    feature_reader.plot_subject_grid_ch_performance(performance_dict=performance_dict, plt_grid=True)
