from scipy import signal
import numpy as np

class NM_Coherence:
    
    def __init__(self, fs, window, fbands, fband_names, ch_1_name, ch_2_name, ch_1_idx, ch_2_idx, coh, icoh) -> None:
        self.fs = fs
        self.window = window
        self.Pxx = None
        self.Pyy = None
        self.Pxy = None
        self.f = None
        self.coh = coh
        self.icoh = icoh
        self.coh_val = None
        self.icoh_val = None
        self.ch_1 = ch_1_name
        self.ch_2 = ch_2_name
        self.ch_1_idx = ch_1_idx
        self.ch_2_idx = ch_2_idx
        self.fbands = fbands  # list of lists, e.g. [[10, 15], [15, 20]]
        self.fband_names = fband_names
        pass

    def get_coh(self, features_, x, y):
        self.f, self.Pxx = signal.welch(x, self.fs, self.window)
        self.Pyy = signal.welch(y, self.fs, self.window)[1]
        self.Pxy = signal.csd(x, y, self.fs, self.window)[1]
        
        if self.coh is True:
            self.coh_val = np.abs(self.Pxy**2)/(self.Pxx*self.Pyy)
        if self.icoh is True:
            self.icoh_val = np.array(self.Pxy/(self.Pxx*self.Pyy)).imag
        
        for idx, fband in enumerate(self.fbands):
            if self.coh is True:
                feature_calc = np.mean(self.coh_val[np.bitwise_and(self.f>fband[0],
                                                                          self.f<fband[1])])
                feature_name = '_'.join(["coh", self.ch_1, "to", self.ch_2, self.fband_names[idx]])
                features_[feature_name] = feature_calc
            if self.icoh is True:
                feature_calc = np.mean(self.icoh_val[np.bitwise_and(self.f>fband[0],
                                                                          self.f<fband[1])])
                feature_name = '_'.join(["icoh", self.ch_1, "to", self.ch_2, self.fband_names[idx]])
                features_[feature_name] = feature_calc
        return features_
