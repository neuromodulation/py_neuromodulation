from numpy import empty_like, mean, ndarray, where
import pandas as pd


class RT_rereference:
    def __init__(self, df: pd.DataFrame, split_data: bool = False) -> None:
        """Initialize real-time rereference information.

        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing information about rereferencing, as
            specified in nm_channels.csv.
        split_data : bool, default: False
            If set to True, the rereferenced data will be returned split
            into cortex and subcortex. Defaults to False.
        """

        self.ch_names = list(df["name"])
        self.refs = df["rereference"]
        self.cortex_used = where((df.type == "ecog") & (df.used == 1))[0]
        self.cortex_good = where((df.type == "ecog") & (df.status == "good"))[
            0
        ]
        self.subcortex_used = where(
            df.type.isin(("seeg", "dbs", "lfp")) & (df.used == 1)
        )[0]
        self.subcortex_good = where(
            df.type.isin(("seeg", "dbs", "lfp")) & (df.status == "good")
        )[0]
        self.to_ref_idx = where(
            ~df.type.isin(("seeg", "dbs", "lfp", "ecog")) | (df.used == 0)
        )

        self.split_data = split_data

    def rereference(self, ieeg_batch: ndarray) -> ndarray:

        """Rereference data according to the initialized RT_rereference class.

        Args:
            ieeg_batch (numpy ndarray) :
                shape(n_channels, n_samples) - data to be rereferenced.

        Raises:
            ValueError: rereferencing using undefined channel
            ValueError: rereferencing according to same channel

        Returns:
            reref_data (numpy ndarray): rereferenced data
        """

        data_subcortex = ieeg_batch[self.subcortex_used]
        new_data_subcortex = empty_like(data_subcortex)
        for i, idx in enumerate(self.subcortex_used):
            elec_channel = self.subcortex_used == idx
            ch = data_subcortex[elec_channel, :]
            if self.refs[idx] in ["none", "None"] or pd.isnull(self.refs[idx]):
                new_data_subcortex[i] = ch
            elif self.refs[idx] == "average":
                av = mean(
                    data_subcortex[self.subcortex_good != idx, :], axis=0
                )
                new_data_subcortex[i] = ch - av
            else:
                index = []
                ref_channels = self.refs[idx].split("&")
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in self.ch_names:
                        raise ValueError(
                            "One or more of the reference channels are not "
                            "part of the recording channels."
                        )
                    index.append(self.ch_names.index(ref_channels[j]))
                new_data_subcortex[i] = ch - mean(ieeg_batch[index, :], axis=0)

        data_cortex = ieeg_batch[self.cortex_used]
        new_data_cortex = empty_like(data_cortex)
        for i, idx in enumerate(self.cortex_used):
            elec_channel = self.cortex_used == idx
            ch = data_cortex[elec_channel, :]
            if self.refs[idx] == "none" or pd.isnull(self.refs[idx]):
                new_data_cortex[i] = ch
            elif self.refs[idx] == "average":
                av = mean(data_cortex[self.cortex_good != idx, :], axis=0)
                new_data_cortex[i] = ch - av
            else:
                index = []
                ref_channels = self.refs[idx].split("+")
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in self.ch_names:
                        raise ValueError(
                            "One or more of the reference "
                            "channels are not part of the "
                            "recorded channels."
                        )
                    if ref_channels[j] == self.ch_names[idx]:
                        raise ValueError(
                            "You cannot rereference to the same channel."
                        )
                    index.append(self.ch_names.index(ref_channels[j]))
                new_data_cortex[i] = ch - mean(ieeg_batch[index, :], axis=0)

        if self.split_data:
            return new_data_cortex, new_data_subcortex
        else:
            reref_data = empty_like(ieeg_batch)
            reref_data[self.subcortex_used, :] = new_data_subcortex
            reref_data[self.cortex_used, :] = new_data_cortex
            reref_data[self.to_ref_idx] = ieeg_batch[self.to_ref_idx]
            return reref_data
