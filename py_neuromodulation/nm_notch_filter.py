from mne.filter import notch_filter as mne_notchfilter
from numpy import arange, floor

def notch_filter(data, fs, line_noise=50):

    freqs = arange(
        line_noise, int((floor(fs/2) / line_noise)) * line_noise + line_noise,
        line_noise)

    return mne_notchfilter(
        x=data,
        Fs=fs,
        trans_bandwidth=15,
        freqs=freqs,
        fir_design='firwin',
        notch_widths=3,
        filter_length=data.shape[1]-1,
        verbose=False
    )
