from scipy import signal
import numpy as np


def get_stft_features(features_, fs, data, ch, f_ranges, f_band_names):
    """Get STFT features for different f_ranges

    Parameters
    ----------
    features_ : dict
        feature dictionary
    fs : int/float
        sampling frequency
    data : np.array
        data for single channel, assumed to be one second
    ch : string
        channel name
    f_ranges : list
        list of list with respective frequency band ranges
    f_band_names : list
        list of frequency band names
    """
    # optimized for fs=1000
    f, t, Zxx = signal.stft(data, fs=fs, window='hamming', nperseg=500,
                            boundary='even')
    Z = np.abs(Zxx)
    for idx_fband, f_range in enumerate(f_ranges):

        idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]

        feature_name = '_'.join(
                [ch, 'stft', f_band_names[idx_fband]])
        features_[feature_name] = np.mean(Z[idx_range, :])  # 1. dim: f, 2. dim: t

    return features_
