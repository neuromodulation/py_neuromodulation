from numpy import empty_like, mean, ndarray
from pandas import isnull


class RT_rereference:
    
    def __init__(self, ch_names, refs, to_ref_idx, cortex_idx, subcortex_idx,
                 split_data=False) -> None:
        """Initiatlize real time reference information

        Args:
            ch_names (list): list of all ieeg_batch channel names
            refs (list): reference specification for rereferencing
            to_ref_idx (list): list of ieeg_batch indices that are not
                rereferenced, i.e. copied to "rereferenced" array
            cortex_idx (list): indices of cortical channels
            subcortex_idx (list): indices of subcortical channels
            split_data (bool, optional): If set to True, the rereferenced data
                will be returned split into cortex and subcortex. Defaults to
                False.
        """
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

        reref_data = empty_like(ieeg_batch)
        reref_data[self.to_ref_idx] = ieeg_batch[self.to_ref_idx]

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

                new_data_subcortex[i] = ch - mean(ieeg_batch[index, :],
                                                  axis=0)
        reref_data[self.subcortex_idx, :] = new_data_subcortex

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
        reref_data[self.cortex_idx, :] = new_data_cortex

        if self.split_data:
            return new_data_cortex, new_data_subcortex
        else:
            return reref_data
