"""Re-referencing Module."""
import numpy as np
import pandas as pd


class RT_rereference:
    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize real-time rereference information.

        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing information about rereferencing, as
            specified in nm_channels.csv.


        Raises:
            ValueError: rereferencing using undefined channel
            ValueError: rereferencing according to same channel
        """
        (self.channels_used,) = np.where((df.used == 1))
        (self.channels_not_used,) = np.where((df.used != 1))

        ch_names = df["name"].tolist()
        ch_types = df.type
        refs = df["rereference"]

        type_map = {}
        for ch_type in df.type.unique():
            type_map[ch_type] = np.where(
                (ch_types == ch_type) & (df.status == "good")
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

    def rereference(self, ieeg_batch: np.ndarray) -> np.ndarray:

        """Rereference data according to the initialized RT_rereference class.

        Args:
            ieeg_batch (numpy ndarray) :
                shape(n_channels, n_samples) - data to be rereferenced.

        Returns:
            reref_data (numpy ndarray): rereferenced data
        """

        new_data = []
        for ch_idx in self.channels_used:
            ref_idx = self.ref_map[ch_idx]
            ref_data = ieeg_batch[ref_idx, :]
            new_data_ch = ieeg_batch[ch_idx, :] - np.mean(ref_data, axis=0)
            new_data.append(new_data_ch)

        reref_data = np.empty_like(ieeg_batch)
        reref_data[self.channels_used, :] = np.vstack(new_data)
        reref_data[self.channels_not_used, :] = ieeg_batch[
            self.channels_not_used
        ]

        return reref_data
