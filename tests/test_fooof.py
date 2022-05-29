import os
import sys
import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path

from py_neuromodulation import (
    nm_generator,
    nm_stream_offline,
    nm_IO,
    nm_define_nmchannels,
)

# https://stackoverflow.com/a/10253916/5060208
# despite that pytest needs to be envoked by python: python -m pytest tests/


class TestWrapper:
    def __init__(self):
        """This test function sets a data batch and automatic initialized M1 datafram

        Args:
            PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

        Returns:
            ieeg_batch (np.ndarray): (channels, samples)
            df_M1 (pd Dataframe): auto intialized table for rereferencing
            settings_wrapper (settings.py): settings.json
            fs (float): example sampling frequency
        """

        sub = "testsub"
        ses = "EphysMedOff"
        task = "buttonpress"
        run = 0
        datatype = "ieeg"

        RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

        PATH_RUN = os.path.join(
            os.path.abspath(os.path.join("examples", "data")),
            f"sub-{sub}",
            f"ses-{ses}",
            datatype,
            RUN_NAME,
        )
        PATH_BIDS = os.path.abspath(os.path.join("examples", "data"))
        PATH_OUT = os.path.abspath(
            os.path.join("examples", "data", "derivatives")
        )

        (
            raw,
            self.data,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
        )

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "dbs", "seeg"),
            target_keywords=("SQUARED_ROTATION",),
        )

        self.stream = nm_stream_offline.Stream(
            settings=None,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=True,
        )
        self.stream.reset_settings()
        self.stream.settings["fooof"]["aperiodic"]["exponent"] = True
        self.stream.settings["fooof"]["aperiodic"]["offset"] = True
        self.stream.settings["features"]["fooof"] = True

        self.stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

    def test_fooof_features(self):
        generator = nm_generator.ieeg_raw_generator(
            self.data, self.stream.settings, self.stream.sfreq
        )
        data_batch = next(generator, None)
        feature_series = self.stream.run_analysis.process_data(data_batch)
        # since the settings can define searching for "max_n_peaks" peaks
        # there will be None's in the feature_series
        # with a non successful fit, aperiod features can also be None
        assert feature_series is not None


def test_fooof():
    test_wrapper = TestWrapper()
    test_wrapper.test_fooof_features()
