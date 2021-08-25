from scipy import fft
import numpy as np


def get_fft_features(features_, s, fs, data, KF_dict, ch, f_ranges, f_band_names):
    """Get FFT features for different f_ranges. Data needs to be a batch of 1s length

    Parameters
    ----------
    features_ : dict
        feature dictionary
    s : dict
        settings dict
    fs : int/float
        sampling frequency
    data : np.array
        data for single channel, assumed to be one second
    KF_dict : dict
        Kalmanfilter dictionaries, channel, bandpower and frequency
        band specific
    ch : string
        channel name
    f_ranges : list
        list of list with respective frequency band ranges
    f_band_names : list
        list of frequency band names
    """

    data = data[-int(s["fft_settings"]["windowlength"]*fs/1000):]

    Z = np.abs(fft.rfft(data))
    f = np.arange(0, int(fs/2)+1, 1)
    for idx_fband, f_range in enumerate(f_ranges):
        fband = f_band_names[idx_fband]
        idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
        feature_calc = np.mean(Z[idx_range])

        if s["methods"]["kalman_filter"] is True:
            if fband in s["kalman_filter_settings"]["frequency_bands"]:
                KF_name = '_'.join([ch, 'activity', fband])  # is for stft by default activity
                KF_dict[KF_name].predict()
                KF_dict[KF_name].update(feature_calc)
                feature_calc = KF_dict[KF_name].x[0]  # filtered signal

        feature_name = '_'.join([ch, 'fft', fband])
        features_[feature_name] = feature_calc
    return features_
