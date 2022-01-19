from numpy import diff, sqrt, var
from numpy import log as np_log


def get_bandpower_features(
        features_,
        s,
        seglengths,
        dat_filtered,
        KF_dict,
        ch,
        ch_idx
    ):
    """Calculate features derived from bandpass filtered signal.

    Parameters
    ----------
    features_ : dict
        Estimated feature dictionary
    s : dict
        settings dict
    seglengths : list
        list of time lengts for bandpower feature estimation
    dat_filtered : ndarray, shape (M, n)
        filtered array of M channels of length n
    KF_dict : dict
        Kalmanfilter dictionaries, channel, bandpower and frequency
        band specific
    ch : str
        channel name
    ch_idx : int
        channel index

    Returns
    -------
    features_ : dict
    """
    for idx, f_band in enumerate(s["frequency_ranges"].keys()):
        seglength = seglengths[idx]
        for bp_feature in [k for k, v in s["bandpass_filter_settings"][
                "bandpower_features"].items() if v is True]:
            if bp_feature == "activity":
                if s["bandpass_filter_settings"]["log_transform"]:
                    feature_calc = np_log(
                        var(dat_filtered[ch_idx, idx, -seglength:]))
                else:
                    feature_calc = var(dat_filtered[ch_idx, idx, -seglength:])
            elif bp_feature == "mobility":
                deriv_variance = var(diff(dat_filtered[ch_idx, idx,
                                          -seglength:]))
                feature_calc = sqrt(deriv_variance / var(dat_filtered[ch_idx,
                                                         idx, -seglength:]))
            elif bp_feature == "complexity":
                dat_deriv = diff(dat_filtered[ch_idx, idx, -seglength:])
                deriv_variance = var(dat_deriv)
                mobility = sqrt(deriv_variance / var(dat_filtered[ch_idx,
                                                     idx, -seglength:]))
                dat_deriv_2 = diff(dat_deriv)
                dat_deriv_2_var = var(dat_deriv_2)
                deriv_mobility = sqrt(dat_deriv_2_var / deriv_variance)
                feature_calc = deriv_mobility / mobility
            if (s["methods"]["kalman_filter"] is True) and (bp_feature == "activity"):
                if f_band in s["kalman_filter_settings"]["frequency_bands"]:
                    KF_name = '_'.join([ch, f_band])
                    KF_dict[KF_name].predict()
                    KF_dict[KF_name].update(feature_calc)
                    feature_calc = KF_dict[KF_name].x[0]  # filtered signal

            feature_name = '_'.join(
                [ch, 'bandpass', bp_feature, f_band])
            features_[feature_name] = feature_calc
    return features_
