import numpy as np
import pandas as pd


def rereference(ieeg_batch, ch_names, refs, rest_idx, cortex_idx,
                subcortex_idx, split_data=False):
    """Rereference data.

    This function rereferences data according to the information given in the
    file "*_channels_M1.tsv". This file must be customized by the user before
    running this script.

    Parameters
    ----------
    ieeg_batch : array, shape(n_channels, n_samples)
        the data to be rereferenced.
    ch_names : list
        list of channel names
    refs : list
        list of reference electrodes for rereferencing
    rest_idx : index
        indices of channel that have neither cortical nor subcortical data
    cortex_idx : index
        indices of cortical channels
    subcortex_idx : index
        indices of subcortical channels
    split_data : boolean, optional
        if set to True, the rereferenced data will be returned split into
        cortex and subcortex. The default is False.

    Returns
    -------
    reref_data : array, shape(n_channels, n_samples)
        rereferenced data.

    """

    reref_data = np.empty_like(ieeg_batch)
    reref_data[rest_idx] = ieeg_batch[rest_idx]

    data_subcortex = ieeg_batch[subcortex_idx]
    new_data_subcortex = np.empty_like(data_subcortex)
    for i, idx in enumerate(subcortex_idx):
        elec_channel = subcortex_idx == idx
        ch = data_subcortex[elec_channel, :]
        if refs[idx] in ['none', 'None'] or pd.isnull(refs[idx]):
            new_data_subcortex[i] = ch
        elif refs[idx] == 'average':
            av = np.mean(data_subcortex[subcortex_idx != idx, :], axis=0)
            new_data_subcortex[i] = ch - av
        else:
            index = []
            ref_channels = refs[idx].split('+')
            for j in range(len(ref_channels)):
                if ref_channels[j] not in ch_names:
                    raise ValueError('One or more of the '
                                        'reference channels are not part of '
                                        'the recording channels.')
                index.append(ch_names.index(ref_channels[j]))

            new_data_subcortex[i] = ch - np.mean(ieeg_batch[index, :],
                                                    axis=0)
    reref_data[subcortex_idx, :] = new_data_subcortex

    data_cortex = ieeg_batch[cortex_idx]
    new_data_cortex = np.empty_like(data_cortex)
    for i, idx in enumerate(cortex_idx):
        elec_channel = cortex_idx == idx
        ch = data_cortex[elec_channel, :]
        if refs[idx] == 'none' or pd.isnull(refs[idx]):
            new_data_cortex[i] = ch
        elif refs[idx] == 'average':
            av = np.mean(data_cortex[cortex_idx != idx, :], axis=0)
            new_data_cortex[i] = ch - av
        else:
            index = []
            ref_channels = refs[idx].split('+')
            for j in range(len(ref_channels)):
                if ref_channels[j] not in ch_names:
                    raise ValueError('One or more of the reference '
                                        'channels are not part of the '
                                        'recorded channels.')
                if ref_channels[j] == ch_names[idx]:
                    raise ValueError('You cannot rereference to the same '
                                        'channel.')
                index.append(ch_names.index(ref_channels[j]))
                
            new_data_cortex[i] = ch - np.mean(ieeg_batch[index, :],
                                                axis=0)
    reref_data[cortex_idx, :] = new_data_cortex

    if split_data:
        return new_data_cortex, new_data_subcortex
    else:
        return reref_data
