from mne.filter import notch_filter as mne_notchfilter
from mne import filter
import numpy as np

from numpy import arange, floor


class NotchFilter:

    def __init__(self,
        fs : int,
        line_noise : int = 50,
        notch_widths : int = 3,
        trans_bandwidth : int = 15
        ) -> None:

        # fs -1 due to exactly half of nyquist frequency
        freqs = arange(
            line_noise, int((floor((fs-1)/2) / line_noise)) * line_noise + line_noise,
            line_noise)

        self.fs = fs
        self.freqs = freqs

        filter_length = fs - 1

        # Code is copied from filter.py notch_filter
        if freqs is not None:
            if notch_widths is None:
                notch_widths = freqs / 200.0
            elif np.any(notch_widths < 0):
                raise ValueError('notch_widths must be >= 0')
            else:
                notch_widths = np.atleast_1d(notch_widths)
                if len(notch_widths) == 1:
                    notch_widths = notch_widths[0] * np.ones_like(freqs)
                elif len(notch_widths) != len(freqs):
                    raise ValueError('notch_widths must be None, scalar, or the '
                                    'same length as freqs')

        # Speed this up by computing the fourier coefficients once
        tb_2 = trans_bandwidth / 2.0
        lows = [freq - nw / 2.0 - tb_2
                for freq, nw in zip(freqs, notch_widths)]
        highs = [freq + nw / 2.0 + tb_2
                 for freq, nw in zip(freqs, notch_widths)]

        l_freq = highs
        h_freq = lows
        l_trans_bandwidth = h_trans_bandwidth = tb_2

        self.filt = filter.create_filter(
            data=None,
            sfreq=self.fs,
            l_freq=l_freq,
            h_freq=h_freq,
            filter_length=filter_length,
            l_trans_bandwidth=l_trans_bandwidth,
            h_trans_bandwidth=h_trans_bandwidth,
            method='fir',
            iir_params=None,
            phase='zero',
            fir_window='hamming',
            fir_design='firwin'
        )

    def filter_data(self, data : np.ndarray) -> np.ndarray:

        return filter._overlap_add_filter(
            x=data,
            h=self.filt,
            n_fft=None,
            phase='zero',
            picks=None,
            n_jobs=1,
            copy=True,
            pad='reflect_limited'
        )

def notch_filter_using_mnewrapper(data, fs, line_noise=50):

    # fs -1 due to exactly half of nyquist frequency
    freqs = arange(
        line_noise, int((floor((fs-1)/2) / line_noise)) * line_noise + line_noise,
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
