import json
from os.path import isdir


def test_settings(settings, verbose=True):
    """Test if settings are specified correctly in settings.json

    Parameters
    ----------
    settings: dict, string or path
        settings loaded as dictionary, or string/path pointing to settings.json.
    verbose: boolean
        set to True if feedback is desired.
    Returns
    -------
    None
    """
    if isinstance(settings, dict):
        s = settings
    else:
        with open(settings, encoding='utf-8') as json_file:
            s = json.load(json_file)
        assert (isinstance(s, dict))

    assert (isdir(s["BIDS_path"]))
    assert (isdir(s["out_path"]))
    assert (isinstance(s["sampling_rate_features"], (float, int)))
    assert (isinstance(s["max_dist_cortex"], (float, int)))
    assert (isinstance(s["max_dist_subcortex"], (float, int)))
    assert (isinstance(value, bool) for value in s["methods"].values()), \
        "Methods must be a boolean value."
    assert (any(value is True for value in s["methods"].values())), \
        "Set at least one method to True."
    if s["methods"]["resample_raw"] is True:
        assert (isinstance(s["resample_raw_settings"]["resample_freq"],
                (float, int)))
    if s["methods"]["normalization"] is True:
        assert (isinstance(s["normalization_settings"]["normalization_time"],
                (float, int)))
        assert (s["normalization_settings"]["normalization_method"] in [
            "mean", "median"])
    if s["methods"]["kalman_filter"] is True:
        assert (isinstance(s["kalman_filter_settings"]["Tp"], (float, int)))
        assert (isinstance(s["kalman_filter_settings"]["sigma_w"],
                           (float, int)))
        assert (isinstance(s["kalman_filter_settings"]["sigma_v"],
                           (float, int)))
        assert (s["kalman_filter_settings"]["frequency_bands"]), \
            "No frequency bands specified for Kalman filter."
        assert (isinstance(s["kalman_filter_settings"]["frequency_bands"],
                           list)), "Frequency bands for Kalman filter must " \
                                   "be specified as a list."
        assert (item in s["bandpass_filter_settings"][
            "frequency_ranges"].values() for item in s[
            "kalman_filter_settings"]["frequency_bands"]), \
            "Frequency bands for Kalman filter must also be specified in " \
            "bandpass_filter_settings."
    if s["methods"]["bandpass_filter"] is True:
        assert (isinstance(s["bandpass_filter_settings"]["frequency_ranges"],
                           dict))
        assert (isinstance(value, list) for value in s[
            "bandpass_filter_settings"]["frequency_ranges"].values())
        assert (len(value) == 2 for value in s[
            "bandpass_filter_settings"]["frequency_ranges"].values())
        assert (isinstance(value[0], list) for value in s[
            "bandpass_filter_settings"]["frequency_ranges"].values())
        assert (len(value[0]) == 2 for value in s[
            "bandpass_filter_settings"]["frequency_ranges"].values())
        assert (isinstance(value[1], (float, int)) for value in s[
            "bandpass_filter_settings"]["frequency_ranges"].values())
        assert (isinstance(value, bool) for value in s[
            "bandpass_filter_settings"]["bandpower_features"].values())
        assert (any(value is True for value in s["bandpass_filter_settings"][
            "bandpower_features"].values())), "Set at least one " \
                                              "bandpower_feature to True."
    if s["methods"]["sharpwave_analysis"] is True:
        assert (isinstance(s["sharpwave_analysis_settings"][
                               "filter_low_cutoff"], (int, float)))
        assert (isinstance(s["sharpwave_analysis_settings"][
                               "filter_high_cutoff"], (int, float)))
        assert (s["sharpwave_analysis_settings"]["filter_high_cutoff"] >
                s["sharpwave_analysis_settings"]["filter_low_cutoff"])
    if s["methods"]["pdc"] is True:
        assert (isinstance(s["pdc_settings"]["frequency_ranges"], dict))
        assert (key in s["bandpass_filter_settings"][
            "frequency_ranges"].values() for key in s["pdc_settings"][
            "frequency_ranges"].keys()), \
            "Frequency bands for PDC must also be specified in " \
            "bandpass_filter_settings."
        assert (isinstance(value, list) for value in s["pdc_settings"][
            "frequency_ranges"].values()), "Channels for PDC must be " \
                                           "specified as a list."
        assert (isinstance(value, list) for value in s[
            "pdc_settings"]["frequency_ranges"].values())
        assert (isinstance(s["pdc_settings"]["model_order"], (str, int))), \
            "Model order in PDC settings must be either an integer or \"auto\"."
        if isinstance(s["pdc_settings"]["model_order"], int):
            assert (s["pdc_settings"]["model_order"] == 'auto'), \
                "Model order in PDC settings must be either an integer " \
                "or \"auto\"."
            assert (isinstance(s["pdc_settings"]["max_order"], int)), \
                "Maximum order in PDC settings must be an integer."
        assert (isinstance(s["pdc_settings"]["num_fft"], (str, int))), \
            "mum_fft in PDC settings must be either an integer or \"auto\"."
    if s["methods"]["dtf"] is True:
        assert (isinstance(s["dtf_settings"]["frequency_ranges"], dict))
        assert (key in s["bandpass_filter_settings"][
            "frequency_ranges"].values() for key in s["dtf_settings"][
            "frequency_ranges"].keys()), \
            "Frequency bands for DTF must also be specified in " \
            "bandpass_filter_settings."
        assert (isinstance(value, list) for value in s["dtf_settings"][
            "frequency_ranges"].values()), "Channels for DTF must be " \
                                           "specified as a list."
        assert (isinstance(value, list) for value in s[
            "pdc_settings"]["frequency_ranges"].values())
        assert (isinstance(s["dtf_settings"]["model_order"], (str, int))), \
            "Model order in DTF settings must be either an integer or \"auto\"."
        if isinstance(s["dtf_settings"]["model_order"], int):
            assert (s["dtf_settings"]["model_order"] == 'auto'), \
                "Model order in DTF settings must be either an integer " \
                "or \"auto\"."
            assert (isinstance(s["dtf_settings"]["max_order"], int)), \
                "Maximum order in DTF settings must be an integer."
        assert (isinstance(s["dtf_settings"]["num_fft"], (str, int))), \
            "mum_fft in DTF settings must be either an integer or \"auto\"."
    if verbose:
        print("No Error occurred when checking the settings.")
    return
