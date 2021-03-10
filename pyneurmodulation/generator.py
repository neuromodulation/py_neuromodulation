

def ieeg_raw_generator(ieeg_raw, df_M1, settings, fs):
    """
    This generator function mimics online data acquisition. 
    The df_M1 selected raw channels are iteratively sampled with fs.  
    Args:
        ieeg_raw (np array): shape (channels, time)
        fs (float): 
        fs_new (float): new resampled frequency 
        offset_start (int): size of highest segmenth length, needs to be skipped at the start to have same feature size
    Yields:
        np.array: new batch for run function of full segment length shape
    """

    cnt_fsnew = 0
    offset_start = int(settings["bandpass_filter_settings"]["segment_lengths"][0] * fs)
    fs_new = settings["resampling_rate"]
    used_idx = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index
    
    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew +=1
            continue
        
        cnt_fsnew +=1
        if cnt_fsnew >= (fs/fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[used_idx,cnt-offset_start:cnt]
