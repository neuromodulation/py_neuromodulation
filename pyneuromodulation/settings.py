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
        assert (isinstance(s["kalman_filter_settings"]["frequency_bands"],
                           list)), "Frequency bands for Kalman filter must " \
                                   "be specified as a list."
        assert (s["kalman_filter_settings"]["frequency_bands"]), \
            "No frequency bands specified for Kalman filter."
    if s["methods"]["bandpass_filter"] is True:
        keys = ["feature_labels", "frequency_ranges", "segment_lengths"]
        lists = [s["bandpass_filter_settings"][key] for key in keys]
        assert (isinstance(lis, list) for lis in lists)
        assert (len(lists[0]) == len(lists[1]) == len(lists[2]))
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
    if verbose:
        print("No Error occurred when checking the settings.")
    return
