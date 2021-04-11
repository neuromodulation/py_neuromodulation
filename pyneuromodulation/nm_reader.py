import os
import json
import numpy as np
from scipy.stats import zscore
import pandas as pd

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
                    f_files.append(os.path.basename(dirpath))
        return f_files

    def read_settings(self, feature_file) -> None:
        with open(os.path.join(self.feature_path, feature_file,
                               feature_file+"_SETTINGS.json")) as f:
            self.settings = json.load(f)
        return self.settings

    def read_M1(self, feature_file) -> None:
        self.df_M1 = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                 feature_file+"_DF_M1.csv"), header=0)
        return self.df_M1

    def read_file(self, feature_file) -> None:
        self.features = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                    feature_file+"_FEATURES.csv"), header=0)

    def read_channel_data(self, ch_name) -> None:
        self.ch_name = ch_name
        self.feature_ch_cols = [i for i in list(self.features.columns) if ch_name in i]
        self.feature_ch = self.features[self.feature_ch_cols]
        return self.feature_ch

    def read_label(self, label_name) -> None:
        self.label_name = label_name
        self.label = self.features[label_name]
        return self.label

    def get_epochs(self, data, y_, epoch_len, sfreq, threshold=0):
        """Return epoched data.

        Parameters
        ----------
        data : np.ndarray
            array of extracted features of shape (n_samples, n_channels, n_features)
        y_ : np.ndarray
            array of labels e.g. ones for movement and zeros for no movement or baseline corr. rotameter data
        sfreq : int/float
            sampling frequency of data
        epoch_len : int
            length of epoch in seconds
        threshold : int/float
            (Optional) threshold to be used for identifying events (default=0 for y_tr with only ones
             and zeros)

        Returns
        -------
        epoch_ np.ndarray
            array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr np.ndarray
            array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)
        ind_mov = np.where(np.diff(np.array(y_ > threshold)*1) == 1)[0]
        low_limit = ind_mov > epoch_lim/2
        up_limit = ind_mov < y_.shape[0]-epoch_lim/2
        ind_mov = ind_mov[low_limit & up_limit]
        epoch_ = np.zeros([ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]])
        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])
        for idx, i in enumerate(ind_mov):
            epoch_[idx, :, :, :] = data[i-epoch_lim//2:i + epoch_lim // 2, :, :]
            y_arr[idx, :] = y_[i-epoch_lim//2:i+epoch_lim//2]
        return epoch_, y_arr

    def get_epochs_ch(self, epoch_len, sfreq, threshold):

        self.epoch_len = epoch_len
        self.sfreq = sfreq
        self.threshold = threshold

        X = np.array(self.feature_ch)
        np.expand_dims(X, axis=1)

        X_epoch, y_epoch = self.get_epochs(data=np.expand_dims(np.array(self.feature_ch), axis=1),
                                           y_=self.label, epoch_len=epoch_len,
                                           sfreq=sfreq, threshold=threshold)
        self.X_epoch = X_epoch
        self.y_epoch = y_epoch
        return X_epoch, y_epoch

    def plot_corr_matrix(self, feature_file):

        feature_col_name = [i for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(7, 7))
        corr = self.feature_ch.corr()
        sns.heatmap(corr,
                    xticklabels=feature_col_name,
                    yticklabels=feature_col_name)
        plt.title("Features channel: "+str(self.ch_name))
        PATH_save = os.path.join(self.feature_path, feature_file,
                                 "Features_corr_matr_ch_"+str(self.ch_name)+".png")
        # axes ticks might be too messy
        plt.xticks([])
        plt.yticks([])
        plt.savefig(PATH_save)
        # plt.show()

    def plot_epochs_avg(self, feature_file):

        feature_col_name = [i for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(6, 6))
        plt.imshow(zscore(np.mean(np.squeeze(self.X_epoch), axis=0), axis=0).T, aspect='auto')
        # plt.yticks(np.arange(0, len(feature_col_name), 1), feature_col_name)
        plt.xticks(np.arange(0, self.X_epoch.shape[1], 1),
                   np.round(np.arange(-self.epoch_len/2, self.epoch_len/2, 1/self.sfreq), 2), rotation=90)
        plt.xlabel("Time [s]")
        plt.title("Movement aligned features channel: "+str(self.ch_name))
        # feature axes ticks might be too messy
        plt.yticks([])
        plt.tight_layout()

        PATH_save = os.path.join(self.feature_path, feature_file,
                                 "MOV_algined_features_ch_"+str(self.ch_name)+".png")
        plt.savefig(PATH_save)
        # plt.show()
