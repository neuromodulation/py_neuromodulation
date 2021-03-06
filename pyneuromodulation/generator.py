from numpy import ceil


def ieeg_raw_generator(ieeg_raw, settings):
    """
    This generator function mimics online data acquisition.
    The df_M1 selected raw channels are iteratively sampled with fs.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
    Returns
    -------
        np.array: new batch for run function of full segment length shape
    """

    cnt_fsnew = 0
    offset_time = max([value[1] for value in settings[
        "bandpass_filter_settings"]["frequency_ranges"].values()])
    offset_start = ceil(offset_time/1000 * settings["fs"]).astype(int)
    fs_new = settings["sampling_rate_features"]
    fs = settings["fs"]

    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew += 1
            continue

        cnt_fsnew += 1
        if cnt_fsnew >= (fs/fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[:, cnt-offset_start:cnt]
