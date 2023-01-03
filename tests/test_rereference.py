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

        self.nm_channels = nm_define_nmchannels.set_channels(
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
            nm_channels=self.nm_channels,
            path_grids=None,
            verbose=True,
        )
        self.stream.set_settings_fast_compute()  # includes rereference features

        self.stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        generator = nm_generator.raw_data_generator(
            self.data, self.stream.settings, self.stream.sfreq
        )
        self.ieeg_batch = next(generator, None)

    def test_rereference(self):
        """
        Args:
            ref_here (RT_rereference): Rereference initialized object
            ieeg_batch (np.ndarray): sample data
            df_M1 (pd.Dataframe): rereferencing dataframe
        """
        ref_dat = self.stream.rereference.rereference(self.ieeg_batch)

        print("Testing channels which are used but not rereferenced.")
        for no_ref_idx in np.where(
            (self.nm_channels.rereference == "None") & self.nm_channels.used
            == 1
        )[0]:
            assert_array_equal(
                ref_dat[no_ref_idx, :], self.ieeg_batch[no_ref_idx, :]
            )

        print("Testing ECOG average reference.")
        for ecog_ch_idx in np.where(
            (self.nm_channels["type"] == "ecog")
            & (self.nm_channels.rereference == "average")
        )[0]:
            assert_array_equal(
                ref_dat[ecog_ch_idx, :],
                self.ieeg_batch[ecog_ch_idx, :]
                - self.ieeg_batch[
                    (self.nm_channels["type"] == "ecog")
                    & (self.nm_channels.index != ecog_ch_idx)
                ].mean(axis=0),
            )

        print("Testing bipolar reference.")
        for bp_reref_idx in [
            ch_idx
            for ch_idx, ch in enumerate(self.nm_channels.rereference)
            if ch in list(self.nm_channels.name)
        ]:
            # bp_reref_idx is the channel index of the rereference anode
            # referenced_bp_channel is the channel index which is the rereference cathode
            referenced_bp_channel = np.where(
                self.nm_channels.iloc[bp_reref_idx]["rereference"]
                == self.nm_channels.name
            )[0][0]
            assert_array_equal(
                ref_dat[bp_reref_idx, :],
                self.ieeg_batch[bp_reref_idx, :]
                - self.ieeg_batch[referenced_bp_channel, :],
            )
