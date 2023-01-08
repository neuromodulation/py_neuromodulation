import math
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from py_neuromodulation.nm_rereference import ReReferencer
from py_neuromodulation import (
    nm_generator,
    nm_settings,
    nm_IO,
    nm_define_nmchannels,
)


class ReReferencerOld:
    def __init__(
        self,
        sfreq: int | float,
        nm_channels: pd.DataFrame,
    ) -> None:
        """Initialize real-time rereference information.

        Parameters
        ----------
        nm_channels : Pandas DataFrame
            Dataframe containing information about rereferencing, as
            specified in nm_channels.csv.


        Raises:
            ValueError: rereferencing using undefined channel
            ValueError: rereferencing according to same channel
        """
        (self.channels_used,) = np.where((nm_channels.used == 1))
        (self.channels_not_used,) = np.where((nm_channels.used != 1))

        ch_names = nm_channels["name"].tolist()
        ch_types = nm_channels.type
        refs = nm_channels["rereference"]

        type_map = {}
        for ch_type in nm_channels.type.unique():
            type_map[ch_type] = np.where(
                (ch_types == ch_type) & (nm_channels.status == "good")
            )[0]

        self.ref_map = {}
        for ch_idx in self.channels_used:
            ref = refs[ch_idx]
            if ref.lower() == "none" or pd.isnull(ref):
                ref_idx = None
            elif ref == "average":
                ch_type = ch_types[ch_idx]
                ref_idx = type_map[ch_type][type_map[ch_type] != ch_idx]
            else:
                ref_idx = []
                ref_channels = ref.split("&")
                for ref_chan in ref_channels:
                    if ref_chan not in ch_names:
                        raise ValueError(
                            "One or more of the reference channels are not"
                            " part of the recording channels. First missing"
                            f" channel: {ref_chan}."
                        )
                    if ref_chan == ch_names[ch_idx]:
                        raise ValueError(
                            "You cannot rereference to the same channel."
                            f" Channel: {ref_chan}."
                        )
                    ref_idx.append(ch_names.index(ref_chan))
            self.ref_map[ch_idx] = ref_idx

    def process(self, data: np.ndarray) -> np.ndarray:

        """Rereference data according to the initialized ReReferencer class.

        Args:
            ieeg_batch (numpy ndarray) :
                shape(n_channels, n_samples) - data to be rereferenced.

        Returns:
            reref_data (numpy ndarray): rereferenced data
        """

        new_data = []
        for ch_idx in self.channels_used:
            ref_idx = self.ref_map[ch_idx]
            if ref_idx is None:
                new_data_ch = data[ch_idx, :]
            else:
                ref_data = data[ref_idx, :]
                new_data_ch = data[ch_idx, :] - np.mean(ref_data, axis=0)
            new_data.append(new_data_ch)

        reref_data = np.empty_like(data)
        reref_data[self.channels_used, :] = np.vstack(new_data)
        reref_data[self.channels_not_used, :] = data[self.channels_not_used]

        return reref_data


class TestReReference:
    def setUp(self) -> None:
        """This test function sets a data batch and automatic initialized M1 datafram

        Args:
            PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

        Returns:
            ieeg_batch (np.ndarray): (channels, samples)
            df_M1 (pd Dataframe): auto intialized table for rereferencing
            settings_wrapper (settings.py): settings.json
            fs (float): example sampling frequency
        """
        sub = "000"
        ses = "right"
        task = "force"
        run = 3
        datatype = "ieeg"

        # Define run name and access paths in the BIDS format.
        RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{datatype}.vhdr"

        PATH_RUN = os.path.join(
            os.path.abspath(os.path.join("examples", "data")),
            f"sub-{sub}",
            f"ses-{ses}",
            datatype,
            RUN_NAME,
        )
        PATH_BIDS = os.path.abspath(os.path.join("examples", "data"))

        (
            raw,
            data,
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

        settings = nm_settings.get_default_settings()
        settings = nm_settings.set_settings_fast_compute(
            settings
        )  # includes rereference features

        generator = nm_generator.raw_data_generator(
            data, settings, math.floor(sfreq)
        )
        self.data_batch = next(generator, None)
        self.re_referencer = ReReferencer(sfreq, self.nm_channels)
        self.re_referencer_old = ReReferencerOld(sfreq, self.nm_channels)

    def test_rereference(self) -> None:
        """
        Args:
            ref_here (RT_rereference): Rereference initialized object
            ieeg_batch (np.ndarray): sample data
            df_M1 (pd.Dataframe): rereferencing dataframe
        """
        self.setUp()
        ref_dat = self.re_referencer.process(self.data_batch)

        print("Testing channels which are used but not rereferenced.")
        for no_ref_idx in np.where(
            (self.nm_channels.rereference == "None") & self.nm_channels.used
            == 1
        )[0]:
            assert_allclose(
                ref_dat[no_ref_idx, :], self.data_batch[no_ref_idx, :]
            )

        print("Testing ECOG average reference.")
        for ecog_ch_idx in np.where(
            (self.nm_channels["type"] == "ecog")
            & (self.nm_channels.rereference == "average")
        )[0]:
            assert_allclose(
                ref_dat[ecog_ch_idx, :],
                self.data_batch[ecog_ch_idx, :]
                - self.data_batch[
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
            assert_allclose(
                ref_dat[bp_reref_idx, :],
                self.data_batch[bp_reref_idx, :]
                - self.data_batch[referenced_bp_channel, :],
            )

        ref_dat_old = self.re_referencer_old.process(self.data_batch)
        assert_allclose(ref_dat, ref_dat_old, rtol=1e-7, equal_nan=False)
