import os

from numba import jit
import numpy as np
from plotly import express
from pandas import DataFrame
from scipy.signal import decimate, detrend

import mne
import mne_bids
from pybv import write_brainvision


def bids_save_file(raw, bids_path, return_raw=False):
    """Write preloaded data to BrainVision file in BIDS format.

    Parameters
    ----------
    raw : raw MNE object
        The raw MNE object for this function to write
    bids_path : BIDSPath MNE-BIDS object
        The MNE BIDSPath to the file to be overwritten
    return_raw : boolean, optional
        Set to True to return the new raw object that has been written.
        Default is False.
    Returns
    -------
    raw : raw MNE object or None
        The newly written raw object.
    """
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    folder = bids_path.directory
    events, event_id = mne.events_from_annotations(raw)

    # rewrite datafile
    write_brainvision(data=data, sfreq=sfreq, ch_names=ch_names,
                      fname_base='dummy', folder_out=folder)
    source_path = os.path.join(folder, 'dummy' + '.vhdr')
    raw = mne.io.read_raw_brainvision(source_path)
    raw.info['line_freq'] = 50
    remapping_dict = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith('ECOG'):
            remapping_dict[ch_name] = 'ecog'
        elif ch_name.startswith(('LFP', 'STN')):
            remapping_dict[ch_name] = 'seeg'
        elif ch_name.startswith('EMG'):
            remapping_dict[ch_name] = 'emg'
        # mne_bids can't handle both eeg and ieeg channel types in the same data
        elif ch_name.startswith(('EEG', 'CREF')):
            remapping_dict[ch_name] = 'misc'
        elif ch_name.startswith(('MOV', 'ANALOG', 'ROT', 'ACC', 
                                 'AUX', 'X', 'Y', 'Z', 'MISC')):
            remapping_dict[ch_name] = 'misc'
    raw.set_channel_types(remapping_dict)
    mne_bids.write_raw_bids(raw, bids_path, events, event_id, overwrite=True)
    suffixes = ['.eeg', '.vhdr', '.vmrk']
    dummy_files = [os.path.join(folder, 'dummy' + suffix)
                   for suffix in suffixes]
    for dummy_file in dummy_files:
        os.remove(dummy_file)
    # check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    if return_raw is True:
        return raw


def get_emg_rms(raw, emg_ch, window_len, analog_ch, rereference=False):
    """Return root mean square with given window length of raw object. 
    
     Parameters
    ----------
    raw : MNE raw object
        The data to be processed.
    emg_ch : list of str
        The EMG channels to be processed. Must be of length 1 or 2.
    window_len : float | int | array-like of float/int
        Window length(s) for root mean square calculation in milliseconds.
    analog_ch : str | list of str
        The target channel (e.g., rotameter) to be added to output raw object.
    rereference : boolean (optional)
        Set to True if EMG channels should be referenced in a bipolar montage.
        Default is False.

    Returns
    -------
    raw_rms : MNE raw object
        Raw object containing root mean square of windowed signal and target
        channel.
    """
    
    raw_emg = raw.copy().pick(picks=emg_ch).load_data()
    raw_emg.set_channel_types(mapping={name:'eeg' for name in raw_emg.ch_names})
    if rereference:
        raw_emg = mne.set_bipolar_reference(raw_emg,
                                            anode=raw_emg.ch_names[0],
                                            cathode=raw_emg.ch_names[1],
                                            ch_name=['EMG_BIP'])
    raw_emg = raw_emg.filter(l_freq=15, h_freq=500, verbose=False)
    if isinstance(window_len, (int, float)):
        window_len = [window_len]
    data = raw_emg.get_data()[0]
    data_arr = np.empty((len(window_len), len(data)))
    for idx, window in enumerate(window_len):
        data_rms = rms_window_nb(data, window, raw_emg.info['sfreq'])
        data_rms_zx = (data_rms-np.mean(data_rms))/np.std(data_rms)
        data_arr[idx, :] = data_rms_zx
    data_analog = raw.copy().pick(picks=analog_ch).get_data()[0]
    if np.abs(min(data_analog)) > max(data_analog):
        data_analog = data_analog*-1
    data_all = np.vstack((data_arr, data_analog))
    emg_ch_names = ['EMG_RMS_' + str(window) for window in window_len]
    info_rms = mne.create_info(ch_names=emg_ch_names + [analog_ch],
                               ch_types=['emg'] * len(window_len) + ['misc'],
                               sfreq=raw_emg.info['sfreq'])
    raw_rms = mne.io.RawArray(data_all, info_rms)
    raw_rms.info['meas_date'] = raw.info['meas_date']
    raw_rms.info['line_freq'] = raw.info['line_freq']
    raw_rms.set_annotations(raw.annotations)
    return raw_rms
    

def raw_plotly(mne_raw, file_name, t_slice=(), plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates (exports) the (sliced) MNE raw signal as an HTML plotly plot

    Arguments:
        mne_raw: MNE raw object (output of mne.io.read_raw_...)
        file_name: name (and directory) for the exported html file
        t_slice: tuple of `start` and `end` slice (seconds)
            example: `t_slice = (1, 5)` returns the 1s-5s slice
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2 

    returns nothing
    """
    samp_freq = int(mne_raw.info["sfreq"])
    channels_array = np.array(mne_raw.info["ch_names"])
    if t_slice:
        signals_array, time_array = mne_raw[:, t_slice[0]*samp_freq:t_slice[1]*samp_freq]
    else:
        signals_array, time_array = mne_raw[:, :]

    sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name, plot_title=plot_title,
               do_decimate=do_decimate, do_normalize=do_normalize, do_detrend=do_detrend, padding=padding)

@jit(nopython=True)
def rms_window_nb(data, window_len, sfreq):
    """Return root mean square of input signal with given window length.
    
     Parameters
    ----------
    data : array
        The data to be processed. Must be 1-dimensional.
    window_len : float | int
        Window length in milliseconds.
    sfreq : float | int
        Sampling frequency in 1/seconds.

    Returns
    -------
    data_rms
        Root mean square of windowed signal. Same dimension as input signal
    """
    
    half_window_size = int(sfreq * window_len / 1000 / 2)
    data_rms = np.empty_like(data)
    for i in range(len(data)):
        if i == 0 or i == len(data)-1:
            data_rms[i] = np.absolute(data[i])
        elif i < half_window_size:
            new_window_size = i
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-new_window_size:i+new_window_size], 2)))
        elif len(data)-i < half_window_size:
            new_window_size = len(data)-i
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-new_window_size:i+new_window_size], 2)))
        else:
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-half_window_size:i+half_window_size], 2)))
    return data_rms


def rms(data, axis=-1):
    """
    returns the Root Mean Square (RMS) value of data along the given axis
    """
    assert axis < data.ndim, \
        "No {} axis for data with {} dimension!".format(axis, data.ndim)
    if axis < 0:
        return np.sqrt(np.mean(np.square(data)))
    else:
        return np.sqrt(np.mean(np.square(data), axis=axis))


def sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name, plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates (exports) the signals as an HTML plotly plot

    Arguments:
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        file_name: name (and directory) for the exported html file
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2 
    
    returns nothing
    """
    
    time_array = np.squeeze(time_array)
    signals_array = np.squeeze(signals_array)
    channels_array = np.squeeze(channels_array)
    if signals_array.ndim == 1:
        signals_array = signals_array.reshape(1, -1)
    
    assert signals_array.shape[0] == channels_array.shape[0], \
        "signals_array ! channels_array Dimension mismatch!"
    assert signals_array.shape[1] == time_array.shape[0], \
        "signals_array ! time_array Dimension mismatch!"

    if do_decimate:
        decimate_factor = min(10, int(samp_freq / 200))
        signals_array = decimate(signals_array, decimate_factor)
        time_array = decimate(time_array, decimate_factor)
    if do_detrend == "linear" or do_detrend == "constant":
        signals_array = detrend(signals_array, axis= 1, type=do_detrend, overwrite_data=True)
    if do_normalize:
        eps_ = np.finfo(float).eps
        signals_array = signals_array / (rms(signals_array, axis=1).reshape(-1, 1) + eps_)

    offset_value = padding * rms(signals_array)  # RMS value
    signals_array = signals_array + offset_value * (np.arange(len(channels_array)).reshape(-1, 1))

    signals_df = DataFrame(data=signals_array.T, index=time_array, columns=channels_array)

    fig = express.line(signals_df, x=signals_df.index, y=signals_df.columns,
                       line_shape="spline", render_mode="svg",
                       labels=dict(index="Time (s)",
                                   value="(a.u.)",
                                   variable="Channel"), title=plot_title)
    fig.update_layout(yaxis=dict(tickmode='array',
                                 tickvals=offset_value * np.arange(len(channels_array)),
                                 ticktext=channels_array))

    fig.write_html(str(file_name) + ".html")
    
    
@jit(nopython=True)
def threshold_events(data, thresh):
    """Apply threshold to find start and end of events.
    """

    onoff = np.where(data > thresh, 1, 0)
    onoff_diff = np.zeros_like(onoff)
    onoff_diff[1:] = np.diff(onoff)
    index_start = np.where(onoff_diff == 1)[0]
    index_stop = np.where(onoff_diff == -1)[0]
    arr_start = np.stack((index_start, np.zeros_like(index_start), 
                          np.ones_like(index_start)), axis = 1)
    arr_stop = np.stack((index_stop, np.zeros_like(index_stop), 
                         np.ones_like(index_stop)*-1), axis = 1)
    return np.vstack((arr_start, arr_stop))
