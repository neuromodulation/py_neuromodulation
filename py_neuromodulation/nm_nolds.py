import nolds


def get_nolds_features(features_, s, data, ch):

    if s["nolds_features"]["sample_entropy"]:
        features_[f"{ch}_sample_entropy"] = nolds.sampen(data)
    if s["nolds_features"]["correlation_dimension"]:
        features_[f"{ch}_correlation_dimension"] = nolds.corr_dim(
            data, emb_dim=2
        )
    if s["nolds_features"]["lyapunov_exponent"]:
        features_[f"{ch}_lyapunov_exponent"] = nolds.lyap_r(data)
    if s["nolds_features"]["hurst_exponent"]:
        features_[f"{ch}_hurst_exponent"] = nolds.hurst_rs(data)
    if s["nolds_features"]["detrended_fluctutaion_analysis"]:
        features_[f"{ch}_detrended_fluctutaion_analysis"] = nolds.dfa(data)

    return features_
