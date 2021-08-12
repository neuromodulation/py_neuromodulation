from numpy import diff, sqrt, var


def get_hjorth_raw(features_, data_, ch):

    features_['_'.join([ch, 'RawHjorth_Activity'])] = var(data_)
    deriv_variance = var(diff(data_))
    mobility = sqrt(deriv_variance / var(data_))
    features_['_'.join([ch, 'RawHjorth_Mobility'])] = mobility

    dat_deriv_2_var = var(diff(diff(data_)))
    deriv_mobility = sqrt(dat_deriv_2_var / deriv_variance)
    features_['_'.join([ch,
                        'RawHjorth_Complexity'])] = deriv_mobility / mobility

    return features_
