import mne
import numpy as np
from matplotlib import pyplot as plt
import kalmanfilter, filter, sharpwaves, bandpower, hjorth_raw
import pandas as pd
from multiprocessing import Process, Manager
import time

class Features:
    
    def __init__(self, s, fs, line_noise, channels):
        """
        s (dict) : json settings
        """

        self.ch_names = channels
        self.s = s # settings
        self.fs = fs
        self.line_noise = line_noise
        self.seglengths = (self.fs / np.array(s["bandpass_filter_settings"]["segment_lengths"])).astype(int)
        self.KF_dict = {}
        
        if s["methods"]["bandpass_filter"] is True:
            self.filter_fun = filter.calc_band_filters(s["bandpass_filter_settings"]["frequency_ranges"], \
                                                sample_rate=fs, filter_len=fs + 1)
        
        for channel in self.ch_names:
            if s["methods"]["kalman_filter"] is True:
                for bp_feature in s["bandpass_filter_settings"]["bandpower_features"]:
                    if s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is False:
                        continue
                    for fband_idx, f_band in enumerate(s["bandpass_filter_settings"]["feature_labels"]): 
                        if s["kalman_filter_settings"]["frequency_bands"][fband_idx] is True: 
                            self.KF_dict['_'.join([channel, bp_feature, f_band])] = \
                                kalmanfilter.define_KF(s["kalman_filter_settings"]["Tp"], \
                                            s["kalman_filter_settings"]["sigma_w"], \
                                            s["kalman_filter_settings"]["sigma_v"])
        
    def estimate_features(self, data):
        """
        
        Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave characteristics. 
        
        data (np array) : (channels, time)
        
        returns: 
        dat (pd Dataframe) with naming convention: channel_method_feature_(f_band)
        """
        # this is done in a lot of loops unfortunately, 
        # what could be done is to extract the outer channel loop, which could run in parallel 
        
        #manager = Manager()
        #features_ = manager.dict() #features_ = {}
        features_ = dict()
        
        # notch filter data before feature estimation 
        if self.s["methods"]["notch_filter"]:
            data = mne.filter.notch_filter(x=data, Fs=self.fs, trans_bandwidth=7,
                freqs=np.arange(self.line_noise, 4*self.line_noise, self.line_noise),
                fir_design='firwin', verbose=False, notch_widths=3, filter_length=data.shape[1]-1)
        
        if self.s["methods"]["bandpass_filter"]:
            dat_filtered = filter.apply_filter(data.T, self.filter_fun, self.fs) # shape (bands, time)
        else:
            dat_filtered = None

        # mutliprocessing approach
        '''
        job = [Process(target=self.est_ch, args=(features_, ch_idx, ch)) for ch_idx, ch in enumerate(self.ch_names)]
        _ = [p.start() for p in job]
        _ = [p.join() for p in job]
        '''

        #sequential approach
        #for ch_idx, ch in enumerate(self.ch_names):
        for ch_idx in range(len(self.ch_names)):
            ch = self.ch_names[ch_idx]
            features_ = self.est_ch(features_, ch_idx, ch, dat_filtered, data)

        #return dict(features_) # this is necessary for multiprocessing approach 
        return features_
                    
    def est_ch(self, features_, ch_idx, ch, dat_filtered, data):
            
        if self.s["methods"]["bandpass_filter"]:
            features_ = bandpower.get_bandpower_features(features_, self.s, self.seglengths, dat_filtered, self.KF_dict, ch, ch_idx)
        
        if self.s["methods"]["raw_hjorth"]: 
            hjorth_raw.get_hjorth_raw(features_, data[ch_idx,:], ch)
        
        if self.s["methods"]["return_raw"]:
            features_['_'.join([ch,'raw'])] = data[ch_idx, -1] # data subsampling
        
        if self.s["methods"]["sharpwave_analysis"]: 
            #print('time taken for sharpwave estimation')
            #start = time.process_time()
            features_ = sharpwaves.get_sharpwave_features(features_, self.s, self.fs, \
                data[ch_idx,-int(np.ceil(self.fs / self.s["resampling_rate"])):], ch)  # take only last resampling_rate
            #print(time.process_time() - start)
        return features_