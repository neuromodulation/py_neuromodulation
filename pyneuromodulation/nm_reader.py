import os
import json
import numpy as np
from scipy.stats import zscore
from pandas import read_pickle

from matplotlib import pyplot as plt
import seaborn as sns


class NM_Reader:
    
    def __init__(self, feature_path) -> None:
    
        self.feature_path = feature_path
        self.df_M1 = None
        self.settings = None
        self.features = None
        self.feature_ch = None
        
    def get_feature_list(self) -> list:
        f_files = []
        for dirpath, subdirs, files in os.walk(self.feature_path):
            for x in files:
                if "FEATURES" in x:
                    f_files.append(os.path.join(dirpath, x))
        return f_files

    def read_settings(self, feature_file) -> None:
        with open(os.path.join(self.feature_path, feature_file,
                               feature_file+"_SETTINGS.json")) as f:
            self.settings = json.load(f)

    def read_M1(self, feature_file) -> None:
        self.df_M1 = read_pickle(os.path.join(self.feature_path, feature_file,
                                 feature_file+"_DF_M1.p"))
    
    def read_file(self, feature_file) -> None:
        self.features = read_pickle(feature_file)
    
    def read_channel_data(self, ch_name) -> None:
        self.ch_name = ch_name
        self.feature_ch_cols = [i for i in list(self.features.columns) if ch_name in i]
        self.feature_ch = self.features[self.feature_ch_cols]
    
    def read_label(self, label_name) -> None:
        self.label_name = label_name
        self.label = self.features[label_name]
        
    def get_epochs(self, dat_filtered, y_tr, epoch_len, sfreq, threshold=0):
        """Return epoched data.
        Keyword arguments
        -----------------
        dat_filtered (array) : array of extracted features of shape (n_samples, n_channels, n_features)
        y_tr (array) : array of labels e.g. ones for movement and zeros for no movement or baseline corr. rotameter data
        sfreq (int/float) : sampling frequency of data
        epoch_len (int) : length of epoch in seconds
        threshold (int/float) : (Optional) threshold to be used for identifying events (default=0 for y_tr with only ones
        and zeros)
        Returns
        -------
        filtered_epoch (Numpy array) : array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr (Numpy array) : array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)
        ind_mov = np.where(np.diff(np.array(y_tr>threshold)*1) == 1)[0]
        low_limit = ind_mov > epoch_lim/2
        up_limit = ind_mov < y_tr.shape[0]-epoch_lim/2
        ind_mov = ind_mov[low_limit & up_limit]
        filtered_epoch = np.zeros([ind_mov.shape[0], epoch_lim, dat_filtered.shape[1], dat_filtered.shape[2]])
        y_arr = np.zeros([ind_mov.shape[0],int(epoch_lim)])
        for idx, i in enumerate(ind_mov):
            filtered_epoch[idx,:,:,:] = dat_filtered[i-epoch_lim//2:i+epoch_lim//2,:,:]
            y_arr[idx,:] = y_tr[i-epoch_lim//2:i+epoch_lim//2]
        return filtered_epoch, y_arr
    
    def get_epochs_ch(self, epoch_len, sfreq, threshold):
        
        self.epoch_len = epoch_len
        self.sfreq = sfreq
        self.threshold = threshold
        
        X = np.array(self.feature_ch)
        np.expand_dims(X, axis=1)
        
        X_epoch, y_epoch = self.get_epochs(dat_filtered=np.expand_dims(np.array(self.feature_ch), axis=1), \
                                y_tr=self.label, epoch_len=epoch_len, \
                                sfreq=sfreq, threshold=threshold)
        self.X_epoch = X_epoch
        self.y_epoch = y_epoch
        
    
    def plot_corr_matrix(self): 
        
        feature_col_name = [i for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(7,7), dpi=300)
        corr = self.feature_ch.corr()
        sns.heatmap(corr, 
                    xticklabels=feature_col_name,
                    yticklabels=feature_col_name)
        plt.title("Features channel: "+str(self.ch_name))
        plt.savefig("Features_corr_matr_ch"+str(self.ch_name)+".png")
        #plt.show()
    
    def plot_epochs_avg(self):
        
        feature_col_name = [i for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(6,6), dpi=300)
        plt.imshow(zscore(np.mean(np.squeeze(self.X_epoch), axis=0), axis=0).T)
        plt.yticks(np.arange(0, len(feature_col_name), 1), feature_col_name)
        plt.xticks(np.arange(0, self.X_epoch.shape[1], 1), \
                   np.round(np.arange(-self.epoch_len/2, self.epoch_len/2, 1/self.sfreq),2), rotation=90)
        plt.xlabel("Time [s]")
        plt.title("Movement aligned features channel: "+str(self.ch_name))
        plt.savefig("MOV_algined_features_ch_"+str(self.ch_name)+".png")
        #plt.show()
        