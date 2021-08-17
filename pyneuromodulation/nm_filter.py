from numpy import array, convolve, expand_dims, vstack

from mne.filter import create_filter


def calc_band_filters(
        f_ranges, sfreq, filter_length="999ms", l_trans_bandwidth=4,
        h_trans_bandwidth=4, verbose=None):
    """Calculate bandpass filters with adjustable length for given frequency
    ranges.
    This function returns for the given frequency band ranges the filter
    coefficients with length "filter_len".
    Thus the filters can be sequentially used for band power estimation.
    Parameters
    ----------
    f_ranges : list of lists
        frequency ranges.
    sfreq : float
        sampling frequency.
    filter_length : str, optional
        length of the filter. Human readable (e.g."1000ms" or "1s").
        Default is "999ms".
    l_trans_bandwidth : float, optional
        Length of the lower transition band. The default is 4.
    h_trans_bandwidth : float, optional
        Length of the higher transition band. The default is 4.
    Returns
    -------
    filter_bank : ndarray, shape(n_franges, filter length samples)
        filter coefficients
    """
    filter_list = list()
    for f_range in f_ranges:
        h = create_filter(None, sfreq, l_freq=f_range[0], h_freq=f_range[1],
                          fir_design='firwin',
                          l_trans_bandwidth=l_trans_bandwidth,
                          h_trans_bandwidth=h_trans_bandwidth,
                          filter_length=filter_length, verbose=verbose)
        filter_list.append(h)
    filter_bank = vstack(filter_list)
    return filter_bank


def apply_filter(data, filter_bank):
    """Apply previously calculated (bandpass) filters to data.
    Parameters
    ----------
    data : array (n_samples, ) or (n_channels, n_samples)
        segment of data.
    filter_bank : array
        output of calc_band_filters.
    Returns
    -------
    filtered : array
        (n_chan, n_fbands, filter_len) array conatining the filtered signal
        at each freq band, where n_fbands is the number of filter bands used to
        decompose the signal
    """
    if data.ndim == 1:
        filtered = array(
            [convolve(filter_bank[filt, :], data, mode='same')
             for filt in range(filter_bank.shape[0])])
        filtered = expand_dims(filtered, axis=0)
    elif data.ndim == 2:
        filtered = array(
            [[convolve(filter_bank[filt, :], data[chan, :], mode='same')
              for filt in range(filter_bank.shape[0])]
             for chan in range(data.shape[0])])
    return filtered
