from numpy import array, empty_like, mean, ndarray, where
from pandas import isnull


class RT_rereference:
    
    def __init__(self, df, split_data=False) -> None:
        """Initialize real-time rereference information.

        Parameters
        ----------
        df
        split_data
        """
        """Initiatlize real time reference information

        Args:
            df (Pandas DataFrame) : 
                Dataframe containing information about rereferencing, as 
                specified in M1.tsv.
            split_data (bool, optional): 
                If set to True, the rereferenced data will be returned split 
                into cortex and subcortex. Defaults to
                False.
        """

        ch_names = list(df['name'])
        refs = df['rereference']
        cortex_idx, = where(df.type == 'ecog')
        subcortex_idx = array(
            df[(df["type"] == 'seeg')
               | (df['type'] == 'dbs')
               | (df['type'] == 'lfp')].index)
        print(subcortex_idx)
        to_ref_idx = array(df[(df['used'] == 0)].index)

        self.ch_names = ch_names
        self.refs = refs 
        self.to_ref_idx = to_ref_idx
        self.cortex_idx = cortex_idx 
        self.subcortex_idx = subcortex_idx 
        self.split_data = split_data 
        
    def rereference(self, ieeg_batch) -> ndarray:

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

        data_subcortex = ieeg_batch[self.subcortex_idx]
        new_data_subcortex = empty_like(data_subcortex)
        for i, idx in enumerate(self.subcortex_idx):
            elec_channel = self.subcortex_idx == idx
            ch = data_subcortex[elec_channel, :]
            if self.refs[idx] in ['none', 'None'] or isnull(self.refs[idx]):
                new_data_subcortex[i] = ch
            elif self.refs[idx] == 'average':
                av = mean(data_subcortex[self.subcortex_idx != idx, :],
                          axis=0)
                new_data_subcortex[i] = ch - av
            else:
                index = []
                ref_channels = self.refs[idx].split('+')
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in self.ch_names:
                        raise ValueError('One or more of the '
                                         'reference channels are not part of '
                                         'the recording channels.')
                    index.append(self.ch_names.index(ref_channels[j]))

                new_data_subcortex[i] = ch - mean(ieeg_batch[index, :], axis=0)

        data_cortex = ieeg_batch[self.cortex_idx]
        new_data_cortex = empty_like(data_cortex)
        for i, idx in enumerate(self.cortex_idx):
            elec_channel = self.cortex_idx == idx
            ch = data_cortex[elec_channel, :]
            if self.refs[idx] == 'none' or isnull(self.refs[idx]):
                new_data_cortex[i] = ch
            elif self.refs[idx] == 'average':
                av = mean(data_cortex[self.cortex_idx != idx, :], axis=0)
                new_data_cortex[i] = ch - av
            else:
                index = []
                ref_channels = self.refs[idx].split('+')
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in self.ch_names:
                        raise ValueError('One or more of the reference '
                                         'channels are not part of the '
                                         'recorded channels.')
                    if ref_channels[j] == self.ch_names[idx]:
                        raise ValueError('You cannot rereference to the same '
                                         'channel.')
                    index.append(self.ch_names.index(ref_channels[j]))

                new_data_cortex[i] = ch - mean(ieeg_batch[index, :],
                                               axis=0)

        if self.split_data:
            return new_data_cortex, new_data_subcortex
        else:
            reref_data = empty_like(ieeg_batch)
            reref_data[self.to_ref_idx] = ieeg_batch[self.to_ref_idx]
            reref_data[self.subcortex_idx, :] = new_data_subcortex
            reref_data[self.cortex_idx, :] = new_data_cortex
            return reref_data
