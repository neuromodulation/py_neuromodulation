import sys
#from icn_permutation_test import permutationTest
from itertools import chain
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats, signal, sparse
from scipy.sparse.linalg import spsolve
import mne
import mne_bids
import json
import multiprocessing
import cvxpy as cp
from pybv import write_brainvision
#from numba import njit
# from bids import BIDSLayout
# from coordinates_io import BIDS_coord
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, KFold, StratifiedKFold
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, auc, precision_recall_curve, \
    plot_precision_recall_curve

# Default settings
settings = {
    "f_ranges" : [[4, 8], [8, 12], [13, 20], [20, 35], [60, 80], [90, 200]],
    "f_bands" : ['theta', 'alpha', 'low beta', 'high beta', 'low gamma', 'HFA'],
    #"BIDS_path" : Path("/Users/richardkoehler/Documents/Neurology_Data/BIDS Berlin/"),
    "sample_rate" : 4096,
    "var_rolling_window" : 4096,
    "resampling_rate" : 1000,
    #"out_path_folder" : Path("/Users/richardkoehler/Documents/Neurology_Data/MovementPrediction/test/derivatives/mne-p/preproc")
}

def read_BIDS_file(vhdr_file):
    """
    Read one run file from BIDS standard
    :param vhdr_file:
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(vhdr_file)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names

def get_all_vhdr_files(BIDS_path):
    """
    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    vhdr_files = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(".vhdr"):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files

def read_all_vhdr_filenames(BIDS_path):
    """
    :return: files: list of all vhdr file paths in BIDS_path
    """
    layout = BIDSLayout(BIDS_path)
    files = layout.get(extension='vhdr', return_type='filename')
    return files

def get_line_noise(vhdr_file):
    """Given a vhdr file, the path is altered in order to read the JSON description file.
    
    Args:
        vhdr_file ([type]): [description]
    Returns:
        int: Power Line Frequency
    """
    
    json_run_ = vhdr_file[:-4] + 'json'
    with open(json_run_, 'r') as fp:
        json_run_descr = json.load(fp)
    return int(json_run_descr['PowerLineFrequency'])

def get_subject_sess_task_run(vhdr_file):
    """ Given a vhdr filename (as a string) return the including subject, session, task and run.
    
    Args:
        vhdr_file (string): [description]
    Return:
        subject, sess, task, run
    """
    
    subject = vhdr_file[vhdr_file.find('sub-')+4:vhdr_file.find('ses')-1]

    str_sess = vhdr_file[vhdr_file.find('ses'):]
    sess = str_sess[str_sess.find('-')+1:str_sess.find('_')]
      
    str_task = vhdr_file[vhdr_file.find('task'):]
    task = str_task[str_task.find('-')+1:str_task.find('run')-1]
    
    str_run = vhdr_file[vhdr_file.find('run'):]
    run = str_run[str_run.find('-')+1:str_run.find('_')]
    run2 = str_run[str_run.find('-')+1:str_run.find('.')]
  
    return subject, sess, task, min(run,run2) 

def running_z_score(x_filtered, z_score_running_interval):
    """
    :param x_filtered
    :param z_score_running_interval
    :return: z-scored stream wrt consecutive time interval
    """
    x_filtered_zscored = np.zeros([x_filtered.shape[0], x_filtered.shape[1], x_filtered.shape[2] - z_score_running_interval])
    for band in range(x_filtered.shape[0]):
        for ch in range(x_filtered.shape[1]):
            for time in np.arange(z_score_running_interval, x_filtered.shape[2], 1):
                running_mean = np.mean(x_filtered[band, ch, (time - z_score_running_interval):time])
                running_std = np.std(x_filtered[band, ch, (time - z_score_running_interval):time])
                x_filtered_zscored[band, ch, time - z_score_running_interval] =                     (x_filtered[band, ch, time] - running_mean) / running_std
    return x_filtered_zscored

def running_zscore_label(mov_label, z_score_running_interval):
    """
       :param mov_label
       :param z_score_running_interval
       :return: z-scored stream wrt consecutive time interval
    """
    
    mov_label_zscored = np.zeros([mov_label.shape[0], mov_label.shape[1] - z_score_running_interval])
    for ch in range(mov_label.shape[0]):
        for time in np.arange(z_score_running_interval, mov_label.shape[1], 1):
            running_mean = np.mean(mov_label[ch, (time - z_score_running_interval):time])
            running_std = np.std(mov_label[ch, (time - z_score_running_interval):time])
            mov_label_zscored[ch, time - z_score_running_interval] =                 (mov_label[ch, time] - running_mean) / running_std
    return mov_label_zscored

def z_score_offline(x_filtered):
    """
    :param x_filtered
    :return: simple "offline" z-score for quicker analysis
    """
    
    x_filtered_zscored = np.zeros(x_filtered.shape)
    for band in range(x_filtered.shape[0]):
        for ch in range(x_filtered.shape[1]):
            x_filtered_zscored[band, ch, :] = stats.zscore(x_filtered[band, ch, :])
    return x_filtered_zscored

def z_score_offline_label(mov_label):
    """
    :param mov_label
    :return: simple "offline" z-score for quicker analysis
    """
    mov_label_zscored = np.zeros(mov_label.shape)
    for ch in range(mov_label.shape[0]):
        mov_label_zscored[ch, :] = stats.zscore(mov_label[ch, :])
    return mov_label_zscored


def t_f_transform(x, sample_rate, f_ranges, line_noise):
    """Calculate time frequency transform with mne filter function.
    
    """
    
    filtered_x = []

    for f_range in f_ranges:
        if line_noise in np.arange(f_range[0], f_range[1], 1):
            #do line noise filtering

            x = mne.filter.notch_filter(x=x, Fs=sample_rate,
                freqs=np.arange(line_noise, 4*line_noise, line_noise),
                fir_design='firwin', verbose=False, notch_widths=2)

        h = mne.filter.create_filter(x, sample_rate, l_freq=f_range[0], h_freq=f_range[1],                                      fir_design='firwin', verbose=False, l_trans_bandwidth='auto', h_trans_bandwidth='auto')
        filtered_x.append(np.convolve(h, x, mode='same'))
    return np.array(filtered_x)


def transform_channels(bv_raw, line_noise):
    """Calculate t-f-transform for every channel.
    
    :param bv_raw: Raw (channel x time) datastream
    :return: t-f transformed array in shape (len(f_ranges), channels, time)
    """
    
    x_filtered = np.zeros([len(settings["f_ranges"]), bv_raw.shape[0], bv_raw.shape[1]])
    for ch in range(bv_raw.shape[0]):
        x_filtered[:, ch, :] = t_f_transform(bv_raw[ch, :], settings["sample_rate"], settings["f_ranges"], line_noise)
    return x_filtered


def calc_running_var(x_filtered_zscored, mov_label_zscored):
    """Given the filtered and z-scored data, apply a rolling variance window.
    
    :param x_filtered_zscored
    :param mov_label_zscored
    :param var_interval time window in which the variance is acquired
    :return: datastream and movement adapted arrays
    """
    
    var_interval=settings["var_rolling_window"]
    stream_roll = np.array(pd.Series(x_filtered_zscored[0, 0, :]).rolling(window=var_interval).var())
    stream_roll = stream_roll[~np.isnan(stream_roll)]
    time_series_length = stream_roll.shape[0]

    x_filtered_zscored_var = np.zeros([x_filtered_zscored.shape[0], x_filtered_zscored.shape[1], time_series_length])

    for f in range(len(settings["f_ranges"])):
        for ch in range(x_filtered_zscored.shape[1]):
            stream_roll = np.array(pd.Series(x_filtered_zscored[f, ch, :]).rolling(window=var_interval).var())
            if stream_roll[~np.isnan(stream_roll)].shape[0] == 0:
                x_filtered_zscored_var[f, ch, :] = np.zeros(x_filtered_zscored_var[f, ch, :].shape[0])
            else:
                x_filtered_zscored_var[f, ch, :] = stream_roll[~np.isnan(stream_roll)]
    # change the label vector too
    print('Num of samples to be cropped (due to running_var) from events array in case of MatLab import: ', x_filtered_zscored.shape[2]-time_series_length)
    return x_filtered_zscored_var, mov_label_zscored[:, (x_filtered_zscored.shape[2] - time_series_length):]

def resample(vhdr_file, ch_names, x_filtered_zscored, mov_label_zscored):
    """Data and mov vector is resampled, assumption here: all channels have the same sampling sampling frequency.
    
    Args:
        vhdr_file (): [description]
        ch_names ([type]): [description]
        x_filtered_zscored ([type]): [description]
        mov_label_zscored ([type]): [description]
    """

    fs = settings["sample_rate"]
    print('sample rate: ', fs)
    fs_new = settings["resampling_rate"]
    print('resampling rate: ', fs_new)

    dat_points = x_filtered_zscored.shape[2]
    new_num_data_points = int((dat_points/fs)*fs_new)
    dat_resampled = signal.resample(x_filtered_zscored, num=new_num_data_points, axis=2)
    mov_resampled = signal.resample(mov_label_zscored, num=new_num_data_points, axis=1)

    return dat_resampled, mov_resampled

def write_out_raw(vhdr_file, folder_out, session, var_rolling_window=1.0, resampling=True, resampling_rate=100, test_LM=False, write_json=False, normalize=False):
    """
    Multiprocessing "Pool" function to interpolate raw file from vhdr_file write to out_path
    :param vhdr_file: raw .vhdr file
    :param out_path_folder
    var_rolling_window: time in seconds of bandpower window (default 1s)
    """

    subject, sess, task, run = get_subject_sess_task_run(vhdr_file)

    bv_raw, ch_names = read_BIDS_file(vhdr_file)

    ch_file = vhdr_file[:-9] + 'channels.tsv'  # the channel file name has the same path/structure as the vhdr file
    df = pd.read_csv(ch_file, sep="\t")  # read out the dataframes channel names frequency, here implementation: same fs for all channels in one run
    sample_rate = df['sampling_frequency'][0]
    settings["sample_rate"] = sample_rate
    print('Sample Rate: ', settings["sample_rate"])

    settings["var_rolling_window"] = int(settings["sample_rate"]*var_rolling_window)
    print('Var Roll Window: ', settings["var_rolling_window"])

    settings["resampling_rate"] = resampling_rate

    ind_mov = [ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('MOV') or ch.startswith('ANALOG') or ch.startswith('ROT')]

    # approach only indexing ECOG named channels
    #ind_dat = [ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('ECOG') or ch.startswith('ANALOG')]
    ind_dat = np.arange(bv_raw.shape[0])[~np.isin(np.arange(bv_raw.shape[0]), ind_mov)]

    mov_label = bv_raw[ind_mov, :]

    line_noise = get_line_noise(vhdr_file)

    #bug fix for now, since I don't see a way to insert writing the line noise parameter in write_brainvision (pybv) for write_raw_bids (mne_bids)
    if subject != '016':
        line_noise = 50

    x_filtered = transform_channels(bv_raw[ind_dat, :], line_noise)

    if normalize is True:
        # proxy for offline data analysis
        # it might be that there are NaN values due to no data stream...
        x_filtered_zscored, mov_label_zscored = calc_running_var(x_filtered, mov_label)
        x_filtered_zscored = np.nan_to_num(z_score_offline(x_filtered_zscored))
        mov_label_zscored = np.nan_to_num(z_score_offline_label(mov_label_zscored))
        #x_filtered_zscored, mov_label_zscored = calc_running_var(x_filtered_zscored, mov_label_zscored)
        #x_filtered_zscored = np.clip(x_filtered_zscored, -2, 2)

    else:
        x_filtered_zscored, mov_label_zscored = calc_running_var(x_filtered, mov_label)

    if test_LM is True:
        for ch in range(bv_raw[ind_dat, :].shape[0]):
            print('Channel No.: ', ch)
            print('R2-Score Linear Model: ', np.mean(cross_val_score(linear_model.LinearRegression(), x_filtered_zscored.T, mov_label_zscored, cv=5, scoring='r2')))


    if resampling is True:
        x_filtered_zscored, mov_label_zscored = resample(vhdr_file, ch_names, x_filtered_zscored, mov_label_zscored)

    if write_json is True:
        dict_ = {
            #"coords": BIDS_coord.get_coord_from_vhdr(settings["BIDS_path"], vhdr_file),
            "subject": subject,
            "sess": session,
            "task": task,
            "run": run,
            "sampling_rate": settings["resampling_rate"],
            "normalized": normalize,
            "ch_names": ch_names,
            "f_ranges": settings["f_ranges"],
            "f_bands": settings["f_bands"],
            "data": x_filtered_zscored.tolist(),
            "true_movements": mov_label_zscored.tolist()
            }
        outpath_file = os.path.join(folder_out, 'xfzs_' + 'sub-' + subject + '_sess-' + session + '_task-' + task + '_run-' + run + '.json')
        with open(outpath_file, 'w') as fp:
            json.dump(dict_, fp)

    return x_filtered_zscored, mov_label_zscored, settings["resampling_rate"]

def NormalizeData(data):
    minv = np.min(data)
    maxv = np.max(data)
    data_new = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_new, minv, maxv

def DeNormalizeData(data, minv, maxv):

    data_new=(data + minv) * (maxv - minv)
    return data_new
    
def baseline_als(y, lam, p, niter=10):
    """
    Baseline drift correction based on [1]
    Inputs:
       y: row signal to be cleaned (array, numpy array)
       lam: reg. parameter (int)
       p: asymmetric parameter. Value in (0 1).

    Problem to Solve (W + lam*D'*D)z=Wy,
    where W=diag(w), D=second order diff. matrix (linear problem)
    [1] P. H. C. Eilers, H. F. M. Boelens, Baseline correction with asymmetric least squares smoothing,
    Leiden University Medical Centre report, 2005.
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_rope(y, lam=1):
    """
   Baseline drift correction based on [1]
   Inputs:
       y: row signal to be cleaned (array, numpy array)
       lam: reg. parameter (int)
   Problem to Solve min |y-b| + lam*(diff_b)^2, s.t. b<=y

   [1] Xie, Z., Schwartz, O., & Prasad, A. (2018). Decoding of finger
   trajectory from ECoG using deep learning. Journal of neural engineering,
   15(3), 036009.
  """
    b = cp.Variable(y.shape)
    objective = cp.Minimize(cp.norm(y-b, 2)+lam*cp.sum_squares(cp.diff(b, 1)))
    constraints = [b <= y]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS")
    z=b.value #--> baseline

    return z

def baseline_correction(y, method='baseline_rope', param=1e4, thr=1e-1,
                        normalize=True, Decimate=1, Verbose=True):
    """

    Parameters
    ----------
    y : array/np.array
        raw signal to be corrected
    method : string, optional
        two possible method for baseline correction are allowed 'baseline_rope'
        and 'baseline_als'. See documentation of each method. The default is 'baseline_rope'.
    param : number or array of numbers, optional
        parameters needed in each optimization method. If baseline_rope is being used, param
        refers to the regularization parameter. If baseline_als is being used
        param should be a 2-lenght array which first value is the regularization
        parameter and the second is the weigthed value. The default is [1e2, 1e-4].
    thr : number, optional
        threshold value in each small variation between trails could still remains
        after baseline elimination. The default is 1e-1.
    normalize : boolean, optional
        if normalize is True the original signal as well as the output corrected signal
        will be scalled between 0 and 1. The default is True.
    Decimate: number, optinal
        before baseline correction it might be necessary to downsample the original raw
        signal. We recommend to do this step. The default is 1, i.e. no decimation.
    Verbose: boolean, optional
        The default is True.
    Returns
    -------
    y_corrected: signal with baseline correction
    onoff: squared signal useful for onset target evaluation.
    y: original signal
    """
    if Decimate != 1:
        if Verbose:
            print('>>Signal decimation is being done')
        y=signal.decimate(y, Decimate)

    if method == 'baseline_als' and np.size(param)!=2:
        raise ValueError("If baseline_als method is desired, param should be "
                         "a 2 length object")
    if method == 'baseline_rope' and np.size(param)>1:
        raise ValueError("If baseline_rope method is desired, param should be "
                         "a number")

    if method=='baseline_als':
        if Verbose:
            print('>>baseline_als is being used')
        z = baseline_als(y, lam=param[0], p=param[1])
    else:
        if Verbose:
            print('>>baseline_rope is being used')
        z = baseline_rope(y, lam=param)

    #subtract baseline
    y_corrected = y-z

    #normalize
    y_corrected, minv, maxv = NormalizeData(y_corrected)

    #eliminate interferation
    y_corrected[y_corrected<thr] = 0
    #create on-off signal
    onoff=np.zeros(np.size(y_corrected))
    onoff[y_corrected>0] = 1

    if normalize:
        y, Nan, Nan = NormalizeData(y)
    else:
        y_corrected = DeNormalizeData(y_corrected, minv, maxv)
    return y_corrected, onoff, y

def create_events_array(onoff, raw_target_channel, sf):
    """

    Parameters
    ----------
    onoff : array, shape(n_samples)
        onoff array. squared signal. when up it indicates the target taks
        was being done. Output of baseline_correction
    raw_target_channel : array, shape(n_samples2)
        the raw signal which which contains the performed taks.
        Needed to estimate time of the g.
    sf : float
        sampling frequency of the raw_target_channel.
        Needed to estimate the time of the events.
    Returns
    -------
    events : array, shape(n_events, 2)
        All events that were found.
        The first column contains the event time in samples and the
        second column contains the event id.
        1= taks starts, -1=taks stops
    """

    #create time vector
    T=len(raw_target_channel)/sf
    Df=len(raw_target_channel)/len(onoff)
    Af=round(T-Df/sf)

    #time onoff_signal
    t= np.arange(0.0, Af, Df/sf)
    print('tshape= ', t.shape)

    #diff to find up and down times
    onoff_dif=np.diff(onoff)
    #create time info
    index_start=onoff_dif==1
    print('index_start= ', index_start.shape)
    time_start=t[index_start]
    index_stop=onoff_dif==-1
    time_stop=t[index_stop]

    if len(time_stop) > len(time_start):
        if time_stop[0]<time_start[0]:
            time_stop=time_stop[1:]
    else:
        if time_start[-1]>time_stop[-1]:
            time_start=time_start[:-1]

    time_event=np.hstack((time_start, time_stop))
    time_event=np.sort(time_event)


    id_event=np.asarray([1, -1]*len(time_start))

    print('id_event: ', id_event.shape)
    print('time_event: ', time_event.shape)

    events=np.transpose(np.vstack((time_event, id_event ))).astype(int)

    return events

def icn_reref(fpath):
    """Rereference ECoG (common average) and LFP (bipolar) data. Return rereferenced raw instance.

    Keyword arguments:
    fname (string) -- filename including path of file in brainvision format

    Return:
    raw (MNE Raw instance) -- data in FIF format
    """

    # select file in bids format
    raw = mne_bids.read.io.brainvision.read_raw_brainvision(fpath, preload=True)

    # set channeltypes
    chan_type_set = {'ECOG_AT_SM_L_{:1}'.format(n): 'ecog' for n in range(1, 7)}
    chan_type_set.update({'LFP_BS_STN_R_{:1}'.format(n): 'ecog' for n in range(2, 5)})
    chan_type_set.update({'LFP_BS_STN_L_{:1}'.format(n): 'ecog' for n in range(1, 5)})
    chan_type_set.update({'ANALOG_ROT_R_1': 'misc'})
    raw.set_channel_types(mapping=chan_type_set)

    misc_renaming_dict = {'ANALOG_ROT_R_1': 'ROTA'}
    raw.rename_channels(misc_renaming_dict)

    # rereference and rename ECoG channels
    ecog_list = ['ECOG_AT_SM_L_{:1}'.format(n) for n in range(1, 7)]
    raw_ecog_reref = raw.copy().pick(ecog_list).set_eeg_reference(ref_channels='average', projection=False,
                                                                  verbose=False)
    ecog_renaming_dict = {name: name + 'r' for name in raw_ecog_reref.ch_names}
    raw_ecog_reref.rename_channels(ecog_renaming_dict)

    # rereference LFP_STNL8 against LFP_STNL1 and rename
    raw_lfp_reref = raw.copy().pick(['LFP_BS_STN_L_1', 'LFP_BS_STN_L_4']).set_eeg_reference(
        ref_channels=['LFP_BS_STN_L_1'], \
        projection=False, verbose=False)
    lfp_renaming_dict = {name: name + 'r' for name in raw_lfp_reref.ch_names}
    raw_lfp_reref.rename_channels(lfp_renaming_dict)

    # add rereferenced ECoG channels to raw
    raw.add_channels([raw_ecog_reref], force_update_info=True)

    # add rereferenced LFP_STNL8 channel to raw
    raw.add_channels([raw_lfp_reref.pick(['LFP_BS_STN_L_4r'])], force_update_info=True)

    ch_names = sorted(list(ecog_renaming_dict.values()) + list(misc_renaming_dict.values()) + \
                      ['LFP_BS_STN_R_4', 'LFP_BS_STN_L_4r'])
    raw.reorder_channels(ch_names)
    return raw


def get_crop_data(raw, cropmin, cropmax, xlim_min=80000, xlim_max=80000):
    """Plot and crop data, then write brainvision file and return cropped data.

    Keyword arguments
    -----------------
    raw (MNE Raw instance) -- MNE Raw data
    cropmin (integer) -- cropping limit at beginning of recording
    cropmax (integer) -- cropping limit at end of recording
    xlim_min (integer) -- upper limit of "cropmin" plot - adjust if needed (default=80000)
    xlim_max (integer) -- lower limit of "cropmax" plot - adjust if needed (default=80000)

    Returns
    -------
    raw (MNE Raw instance) -- data in FIF format
    data (NumPy array) -- data as numpy array
    """

    data = raw.get_data()
    # crop artefacts at end and beginning of recording after visual inspection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 2))
    # 'cropmin': determine lower cut off for cropping from visual inspection
    ax1.plot(data[-1, :])
    ax1.axvline(cropmin, label='cropmin', color='r')
    ax1.set_title('cropmin')
    ax1.set_xlim(0, xlim_min)
    ax1.legend(loc='upper left')

    # 'cropmax': determine upper cut off for cropping from visual inspection
    ax2.plot(data[-1, :])
    ax2.axvline(cropmax, label='cropmax', color='r')
    ax2.set_title('cropmax')
    ax2.set_xlim(len(data[-1, :]) - xlim_max, len(data[-1, :]))
    ax2.legend(loc='upper left')
    plt.show()

    tmin = cropmin / raw.info['sfreq']
    tmax = cropmax / raw.info['sfreq']
    crop_raw = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)
    return crop_raw


def write_data_to_bv(data, sfreq, ch_names, fname_base, folder_out, events=None):
    """Write preprocessed data to brainvision file.
    """

    fname_base = 'events_crop_relevchan_reref_' + fname_base
    write_brainvision(data=data, sfreq=sfreq, ch_names=ch_names, fname_base=fname_base, folder_out=folder_out, \
                      events=events, resolution=1e-07, unit='uV')
    print("Please copy and modify .json and .tsv files.")
    print("New filename for .json and .tsv files: ", fname_base)
    print("New RecordingDuration (enter into .json file): ", len(crop_data[1]) / crop_raw.info['sfreq'])
    print("New number of channels(enter into .json file): ", crop_raw.info['nchan'])
    print("New channel names and types: ", crop_raw.info['ch_names'])


def retrieve_data(path, fname):
    """Read json file and return processed data.

    """
    subject, session, task, run = get_subject_sess_task_run(fname)
    datafile_path = os.path.join(path, 'xfzs_' + 'sub-' + subject + '_sess-' + session + '_task-' + task + '_run-' + run + '.json')
    with open(datafile_path) as json_file:
        data = json.load(json_file)
    return data

def retrieve_events(path, fname):
    """Read json file and return processed data.

    """
    subject, session, task, run = get_subject_sess_task_run(fname)
    datafile_path = os.path.join(path, 'evs_' + 'xfzs_' + 'sub-' + subject + '_sess-' + session + '_task-' + task + '_run-' + run + '.json')
    with open(datafile_path) as json_file:
        data = json.load(json_file)
    return data


def get_events(mov_raw, method, param, thr, invertrota=True, decimate=1):
    """Extract events from movement data.

    """
    
    if type(mov_raw) == (mne.io.brainvision.brainvision.RawBrainVision or mne.io.edf.edf.RawEDF):
        ind_mov = [ch for ch in mov_raw.info['ch_names'] if ch.startswith('MOV') or ch.startswith('ANALOG') or ch.startswith('ROT')]
        mov_raw = mov_raw.get_data(picks=ind_mov[0])
        print('Channel used: ', ind_mov[0])

    if invertrota is True:
        mov_raw = mov_raw[0,:].T * -1
    else:
        mov_raw = mov_raw[0,:].T
        
    mov_corrected, onoff, mov = baseline_correction(mov_raw, method=method, param=param, \
                                                                     thr=thr, normalize=True, Decimate=decimate, Verbose=True)

    # Create array of events
    events_arr = create_events_array(onoff, mov_raw.T, 1)
    print('Number of detected events: ', len(events_arr[:, 0]) / 2)

    # Plot, evaluate, tweak param+thr in baseline_correction and rerun
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axs[0, 0].plot(mov_raw)
    axs[0, 0].set_title('Raw data')
    axs[0, 1].plot(mov, 'tab:orange')
    axs[0, 1].set_title('mov')
    axs[1, 0].plot(mov_corrected, 'tab:green')
    axs[1, 0].set_title('mov_corrected')
    axs[1, 1].plot(onoff, 'tab:red')
    axs[1, 1].set_title('onoff')
    plt.show()

    # Now plot and check accuracy
    plt.figure(dpi=300, figsize=(7, 5))
    plt.plot(events_arr[0:2, 0], mov_raw[events_arr[0:2, 0]], 'r*')
    plt.plot(mov_raw[:events_arr[2, 0]])
    plt.show()
    
    events_mne = np.array([events_arr[:,0],np.zeros_like(events_arr[:,0]),events_arr[:,1]]).T
    return events_mne


def get_epochs(raw, epoch_arr, event_id, tmin, tmax, plot_mov=True):
    """
    """

    epochs = mne.Epochs(raw=raw, events=epoch_arr, event_id=event_id,  tmin=tmin, tmax=tmax, preload=True)
    if plot_mov is True:
        ind_mov = [ch for ch_idx, ch in enumerate(raw.info['ch_names']) if ch.startswith('MOV') or ch.startswith('ANALOG') or ch.startswith('ROT')]
        epochs.average(picks=ind_mov[0]).plot()
    return epochs


def morlet_tf(epochs, lfreq, hfreq, n_cycles, tmin, tmax, vmin, vmax, baseline=(-5, -4), plot=True, plot_chan=None):
    """
    """

    freqs = np.arange(lfreq, hfreq, 2)
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=False, \
                                          average=True, verbose=None, zero_mean=True)
    if plot is True:
        power.plot(plot_chan, baseline=baseline, mode='zscore', colorbar=True, tmin=-4, tmax=4, \
                   cmap=('viridis', True), vmax=3, vmin=-3)
    return power


def plot_power(power, rows, columns, tmin, tmax, vmin, vmax, size, title, savefig=True):
    """
    """
    plt.style.use('dark_background')
    fig, axs = plt.subplots(rows, columns, sharey=True, figsize=size, dpi=300)
    chnames = power.info['ch_names']
    i = 0
    for r in range(rows):
        for c in range(columns):
            power.plot(chnames[i], baseline=(-5, -4), mode='zscore', colorbar=False, tmin=tmin, tmax=tmax, \
                       cmap=('viridis', True), vmax=vmax, vmin=vmin, show=False, axes=axs[r, c])
            axs[r, c].set_title(chnames[i], fontsize=16)
            i = i + 1
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    if savefig is True:
        fig.savefig(outpath_proc + 'morlet_' + 'sub_' + subject + '_run_' + run + '_sess_' + sess + '.png')
        
def generate_continous_label_array(L, sf, events):
    """
    given and arrray of events, this function returns sample-by-sample
    label information of raw_date
    Parameters
    ----------
    L : float, 
        length (n_samples) of the corresponding signal to labelled
    sf : int, float
        sampling frequency of the raw_data
    events : array, shape(n_events,2)
        All events that were found by the function
        'create_events_array'. 
        The first column contains the event time in samples and the second column contains the event id.
   
    Returns
    -------
    labels : array (n_samples)
        array of ones and zeros.
    """
    
    labels=np.zeros(L)
    groups=np.zeros(L)
    
    mask_start=events[:,-1]==1
    start_event_time=events[mask_start,0]
    mask_stop=events[:,-1]==-1
    stop_event_time=events[mask_stop,0]
    
    group = 0
    for i in range(len(start_event_time)):
        range_up=np.arange(int(np.round(start_event_time[i]*sf)), int(np.round(stop_event_time[i]*sf)))
        labels[range_up]=1
        if i == 0:
            range_group=np.arange(0, int(np.round(stop_event_time[i]*sf)))
        else:
            range_group=np.arange(int(np.round(stop_event_time[i-1]*sf)), int(np.round(stop_event_time[i]*sf)))
        groups[range_group]=group
        group+=1
        
    return labels, groups