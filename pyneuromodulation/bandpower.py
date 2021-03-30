from numpy import diff, sqrt, var


def get_bandpower_features(features_, s, seglengths, dat_filtered, KF_dict, ch,
                           ch_idx):
    for f_band in range(len(s["bandpass_filter_settings"]["frequency_ranges"])):
        seglength = seglengths[f_band]
        for bp_feature in [k for k, v in
                           s["bandpass_filter_settings"][
                               "bandpower_features"].items() if v is True]:
            if bp_feature == "activity":
                feature_calc = var(dat_filtered[ch_idx, f_band, -seglength:])
            elif bp_feature == "mobility":
                deriv_variance = var(diff(dat_filtered[ch_idx, f_band,
                                          -seglength:]))
                feature_calc = sqrt(deriv_variance / var(dat_filtered[ch_idx,
                                                         f_band, -seglength:]))
            elif bp_feature == "complexity":
                dat_deriv = diff(dat_filtered[ch_idx, f_band, -seglength:])
                deriv_variance = var(dat_deriv)
                mobility = sqrt(deriv_variance / var(dat_filtered[ch_idx,
                                                     f_band, -seglength:]))
                dat_deriv_2 = diff(dat_deriv)
                dat_deriv_2_var = var(dat_deriv_2)
                deriv_mobility = sqrt(dat_deriv_2_var / deriv_variance)
                feature_calc = deriv_mobility / mobility
            if s["kalman_filter_settings"]["frequency_bands"][f_band] is True \
                    and s["methods"]["kalman_filter"] is True:
                KF_name = '_'.join([ch, bp_feature, s[
                    "bandpass_filter_settings"]["feature_labels"][f_band]])
                KF_dict[KF_name].predict()
                KF_dict[KF_name].update(feature_calc)
                feature_calc = KF_dict[KF_name].x[0]  # filtered sensor signal
    
            feature_name = '_'.join([ch, 'bandpass', bp_feature,
                                     s["bandpass_filter_settings"][
                                         "feature_labels"][f_band]])
            features_[feature_name] = feature_calc
    return features_
