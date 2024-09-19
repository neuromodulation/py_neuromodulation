"""Re-referencing Module."""

import numpy as np
import pandas as pd

from py_neuromodulation.utils.types import NMPreprocessor


class ReReferencer(NMPreprocessor):
    def __init__(
        self,
        sfreq: float,
        channels: pd.DataFrame,
    ) -> None:
        """Initialize real-time rereference information.

        Parameters
        ----------
        sfreq : float
            Sampling frequency. Is not used, only kept for compatibility.
        channels : Pandas DataFrame
            Dataframe containing information about rereferencing, as
            specified in channels.csv.


        Raises:
            ValueError: rereferencing using undefined channel
            ValueError: rereferencing to same channel
        """

        self.ref_matrix: np.ndarray | None

        channels = channels[channels["used"] == 1].reset_index(drop=True)
        # (channels_used,) = np.where((channels.used == 1))

        ch_names = channels["name"].tolist()

        # no re-referencing is being performed when there is a single channel present only
        if channels.shape[0] in (0, 1):
            self.ref_matrix = None
            return

        ch_types = channels["type"]
        refs = channels["rereference"]

        type_map = {}
        for ch_type in ch_types.unique():
            type_map[ch_type] = np.where(
                (ch_types == ch_type) & (channels["status"] == "good")
            )[0]

        ref_matrix = np.zeros((len(channels), len(channels)))
        for ind in range(len(channels)):
            ref_matrix[ind, ind] = 1
            # if ind not in channels_used:
            #    continue
            ref = refs[ind]
            if ref.lower() == "none" or pd.isnull(ref):
                ref_idx = None
                continue
            if ref.lower() == "average":
                ch_type = ch_types[ind]
                ref_idx = type_map[ch_type][type_map[ch_type] != ind]
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
                    if ref_chan == ch_names[ind]:
                        raise ValueError(
                            "You cannot rereference to the same channel."
                            f" Channel: {ref_chan}."
                        )
                    ref_idx.append(ch_names.index(ref_chan))
            ref_matrix[ind, ref_idx] = -1 / len(ref_idx)
        self.ref_matrix = ref_matrix

    def process(self, data: np.ndarray) -> np.ndarray:
        """Rereference data according to the initialized ReReferencer class.

        Args:
            data (numpy ndarray) :
                shape(n_channels, n_samples) - data to be rereferenced.

        Returns:
            reref_data (numpy ndarray):
                shape(n_channels, n_samples) - rereferenced data
        """
        if self.ref_matrix is not None:
            return self.ref_matrix @ data
        else:
            return data
