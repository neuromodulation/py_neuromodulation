import mne
import numpy as np
import pandas as pd

def rereference(ieeg_batch, df_M1, get_cortex_subcortex=False):
    """
    Rereference data.

    This function rereference data accordindly to the information given in the
    files "*_channels_MI.tsv". This file must be customized by the user before
    running this script.

    Parameters
    ----------
    ieeg_batch : array, shape(n_channels, n_samples)
        the data to be rerefenced.
    df_M1 : data frame
        data frame with the channels configuration description.
    get_cortex_subcortex : boolean, optional
        if set to true, the rereferenced data will be returned as well as the
        data splitted in cortex and subcortex. The default is False.

    Returns
    -------
    new_data : array, shape(n_channels, n_samples)
        rereferenced raw_data.

    """
    channels_name = df_M1['name'].tolist()
    new_data = ieeg_batch.copy()
    # subcortex
    subcortex_exists = any(df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)
                                 & (df_M1["ECOG"] == 0)].index)
    if subcortex_exists:
        index_channels = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0) &
                               (df_M1["ECOG"] == 0)].index
        data_subcortex = ieeg_batch[index_channels]
        new_data_subcortex = data_subcortex.copy()
        idx = 0
        for ii in index_channels:
            elec_channel = index_channels == ii
            ch = data_subcortex[elec_channel, :]
            if df_M1['rereference'][ii] == 'none' or pd.isnull(df_M1['rereference'][ii]):
                continue
            if df_M1['rereference'][ii] == 'average':
                av = np.mean(data_subcortex[index_channels != ii, :], axis=0)
                new_data_subcortex[idx] = ch-av
            else:
                index = []
                ref_channels = df_M1['rereference'][ii].split('+')

                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or maybe more of the ref_channels'
                                         ' are not part of the recording channels.')
                    index.append(channels_name.index(ref_channels[j]))

                new_data_subcortex[idx] = ch - np.mean(ieeg_batch[index, :], axis=0)
            idx = idx + 1
        new_data[index_channels, :] = new_data_subcortex
    else:
        new_data_subcortex = None

    cortex_exists = any(df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0) & (df_M1["ECOG"] == 1)].index)
    if cortex_exists:
        index_channels = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0) &
                               (df_M1["ECOG"] == 1)].index
        data_cortex = ieeg_batch[index_channels]
        new_data_cortex = data_cortex.copy()
        idx = 0
        for i in index_channels:
            elec_channel = index_channels == i
            ch = data_cortex[elec_channel, :]
            if df_M1['rereference'][i] == 'none' or pd.isnull(df_M1['rereference'][i]):
                continue
            if df_M1['rereference'][i] == 'average':
                av = np.mean(data_cortex[index_channels != i, :], axis=0)
                new_data_cortex[idx] = ch-av
            else:
                index = []
                ref_channels = df_M1['rereference'][i].split('+')
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or more of the ref_channels are not part of the recorded channels.')
                    if ref_channels[j] == channels_name[i]:
                        raise ValueError('You cannot rereference to the same channel.')
                    index.append(channels_name.index(ref_channels[j]))
                    
                new_data_cortex[idx] = ch - np.mean(ieeg_batch[index, :],
                                                    axis=0)
            idx = idx+1
        new_data[index_channels, :] = new_data_cortex
    else:
        new_data_cortex = None
    if get_cortex_subcortex:
        return new_data_cortex, new_data_subcortex
    else:
        return new_data
