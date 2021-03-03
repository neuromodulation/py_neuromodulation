import mne 
import numpy as np 

def calc_band_filters(f_ranges, sample_rate, filter_len="1000ms", l_trans_bandwidth=4, h_trans_bandwidth=4):
    """"Calculate bandpass filters with adjustable length for given frequency ranges.

    This function returns for the given frequency band ranges the filter coefficients with length "filter_len".
    Thus the filters can be sequentially used for band power estimation.

    Parameters
    ----------
    f_ranges : list of lists
        frequency ranges.
    sample_rate : float
        sampling frequency.
    filter_len : str, optional
        length of the filter. Human readable (e.g."1000ms" or "1s"). Default is "1000ms"
    l_trans_bandwidth : int/float, optional
        Length of the lower transition band. The default is 4.
    h_trans_bandwidth : int/float, optional
        Length of the higher transition band. The default is 4.

    Returns
    -------
    filter_fun : array
        filter coefficients stored in array of shape (n_franges, filter_len (in samples))
    """
    filter_list = []
    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1], fir_design='firwin',
                                        l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                        filter_length=filter_len)
        filter_list.append(h)
    filter_fun = np.vstack(filter_list)
    return filter_fun

def apply_filter(dat_, filter_fun, fs):
        """Apply previously calculated (bandpass) filters to data.

        Parameters
        ----------
        dat_ : array (n_samples, ) or (n_channels, n_samples)
            segment of data.
        filter_fun : array
            output of calc_band_filters.

        Returns
        -------
        filtered : array
            (n_chan, n_fbands, filter_len) array conatining the filtered signal
            at each freq band, where n_fbands is the number of filter bands used to decompose the signal
        """    
        if dat_.ndim == 1:
            filtered = np.zeros((1, filter_fun.shape[0], fs))
            for filt in range(filter_fun.shape[0]):
                filtered[0, filt, :] = np.convolve(filter_fun[filt,:], dat_)[int(fs-fs/2):int(fs+fs/2)]
        elif dat_.ndim == 2:
            filtered = np.zeros((dat_.shape[0], filter_fun.shape[0], fs))
            for chan in range(dat_.shape[1]):
                for filt in range(filter_fun.shape[0]):
                    filtered[chan, filt, :] = np.convolve(filter_fun[filt, :], \
                                                        dat_[:, chan])[int(fs-fs/2):int(fs+fs/2)] # mode="full"
        return filtered