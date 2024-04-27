import numpy as np
from scipy import fft

from py_neuromodulation import nm_filter


def test_notch_filter_setup():

    # by Nyquist theorem, frequencies are computed up to half sfreq
    for sfreq in [150, 200, 500, 1000]:
        line_noise = 50

        notch_filter = nm_filter.NotchFilter(sfreq, line_noise)

        # the computed filter is saved in self.filter_bank

        data = np.random.random(sfreq)
        filtered_dat = notch_filter.process(data)

        Z_filtered = np.abs(fft.rfft(filtered_dat))
        Z_nonfiltered = np.abs(fft.rfft(data))
        freqs = fft.rfftfreq(sfreq, 1 / sfreq)
        idx = (np.abs(freqs - line_noise)).argmin()

        assert np.mean(Z_filtered[idx - 1 : idx + 1]) < np.mean(
            Z_nonfiltered[idx - 1 : idx + 1]
        ), (
            f"testing notch filter with sampling frequency {line_noise} failed"
            f" for comparison fft power vs no filtering"
        )
