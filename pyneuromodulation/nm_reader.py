import os
import json
import numpy as np
from scipy.stats import zscore
from scipy import io
import pandas as pd
import _pickle as cPickle
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
        self.feature_file = feature_file
        with open(os.path.join(self.feature_path, feature_file,
                               feature_file + "_SETTINGS.json")) as f:
            self.settings = json.load(f)
        return self.settings

    def read_M1(self, feature_file) -> None:
        self.df_M1 = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                 feature_file + "_DF_M1.csv"), header=0)
        return self.df_M1

    def read_features(self, feature_file) -> None:
        self.features = pd.read_csv(os.path.join(self.feature_path, feature_file,
                                    feature_file + "_FEATURES.csv"), header=0)

    def read_channel_data(self, ch_name, read_bp_activity_only=True) -> None:
        self.ch_name = ch_name
        self.feature_ch_cols = [i for i in list(self.features.columns) if ch_name in i]
        if read_bp_activity_only:
            bp_ = [f for f in self.feature_ch_cols if 'bandpass' in f and 'activity' in f]
            self.feature_ch_cols = bp_[::-1]  # flip list s.t. theta band is lowest in subsequent plot
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
        ind_mov = np.where(np.diff(np.array(y_ > threshold) * 1) == 1)[0]
        low_limit = ind_mov > epoch_lim / 2
        up_limit = ind_mov < y_.shape[0] - epoch_lim / 2
        ind_mov = ind_mov[low_limit & up_limit]
        epoch_ = np.zeros([ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]])
        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])
        for idx, i in enumerate(ind_mov):
            epoch_[idx, :, :, :] = data[i - epoch_lim // 2:i + epoch_lim // 2, :, :]
            y_arr[idx, :] = y_[i - epoch_lim // 2:i + epoch_lim // 2]
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

        feature_col_name = [i[len(self.ch_name)+1:] for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(7, 7))
        corr = self.feature_ch.corr()
        sns.heatmap(corr,
                    xticklabels=feature_col_name,
                    yticklabels=feature_col_name)
        plt.title("Features channel: " + str(self.ch_name))
        PATH_save = os.path.join(self.feature_path, feature_file,
                                 "Features_corr_matr_ch_" + str(self.ch_name) + ".png")
        # axes ticks might be too messy
        #plt.xticks([])
        #plt.yticks([])
        plt.savefig(PATH_save, bbox_inches = "tight")
        print("Correlation matrix figure saved to " + str(PATH_save))

    def plot_epochs_avg(self, feature_file):
        
        # cut channel name of for axis + "_" for more dense plot
        feature_col_name = [i[len(self.ch_name)+1:] for i in list(self.feature_ch_cols) if self.ch_name in i]
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.imshow(zscore(np.mean(np.squeeze(self.X_epoch), axis=0), axis=1).T, aspect='auto')
        plt.yticks(np.arange(0, len(feature_col_name), 1), feature_col_name)
        plt.xticks(np.arange(0, self.X_epoch.shape[1], 1),
                   np.round(np.arange(-self.epoch_len / 2, self.epoch_len / 2, 1 / self.sfreq), 2), rotation=90)
        plt.xlabel("Time [s]")
        plt.title("Movement aligned features channel: " + str(self.ch_name))
        # feature axes ticks might be too messy
        #plt.yticks([])

        plt.subplot(212)
        for i in range(self.y_epoch.shape[0]):
            plt.plot(self.y_epoch[i,:], color="black", alpha=0.4)
        plt.plot(self.y_epoch.mean(axis=0), color="black", alpha=1, linewidth=3.0, label="mean target")
        plt.legend()
        plt.ylabel("target")
        plt.title(self.label_name)
        plt.xticks(np.arange(0, self.X_epoch.shape[1], 1),
                   np.round(np.arange(-self.epoch_len / 2, self.epoch_len / 2, 1 / self.sfreq), 2), rotation=90)
        plt.xlabel("Time [s]")
        plt.tight_layout()

        PATH_save = os.path.join(self.feature_path, feature_file,
                                 "MOV_algined_features_ch_" + str(self.ch_name) + ".png")
        plt.savefig(PATH_save, bbox_inches = "tight")
        # plt.show()
        
        print("Feature epoch average figure saved to: " + str(PATH_save))
    
    def read_ML_estimations(self):
        """Read estimated ML outputs

        Returns
        -------
        nm_decode object
        """
        PATH_ML_ = os.path.join(self.feature_path, self.feature_file, self.feature_file + "_ML_RES.p")
        try:
            with open(PATH_ML_, 'rb') as input:  # Overwrites any existing file.
                ML_est = cPickle.load(input)
        except FileNotFoundError:
            print("no _ML file computed")
            return None
        return ML_est

    def read_run_analyzer(self):
        """Read run_analysis outputs. If target was set to true, a corresponding column was added to feature_arr
        dataframe. 
        Returns
        -------
        run_analysis object
        """
        PATH_ML_ = os.path.join(self.feature_path, self.feature_file, self.feature_file + "_run_analysis.p")
        with open(PATH_ML_, 'rb') as input:  # Overwrites any existing file.
            self.run_analysis = cPickle.load(input)
        return self.run_analysis

    def read_plot_modules(self, PATH_PLOT=os.path.join(os.pardir, 'plots')):
        """Read required .mat files for plotting

        Parameters
        ----------
        PATH_PLOT : regexp, optional
            path to plotting files, by default
        """
        
        self.faces = io.loadmat(os.path.join(PATH_PLOT, 'faces.mat'))
        self.vertices = io.loadmat(os.path.join(PATH_PLOT, 'Vertices.mat'))
        self.grid = io.loadmat(os.path.join(PATH_PLOT, 'grid.mat'))['grid']
        self.stn_surf = io.loadmat(os.path.join(PATH_PLOT, 'STN_surf.mat'))
        self.x_ver = self.stn_surf['vertices'][::2,0]
        self.y_ver = self.stn_surf['vertices'][::2,1]
        self.x_ecog = self.vertices['Vertices'][::1,0]
        self.y_ecog = self.vertices['Vertices'][::1,1]
        self.z_ecog = self.vertices['Vertices'][::1,2]
        self.x_stn = self.stn_surf['vertices'][::1,0]
        self.y_stn = self.stn_surf['vertices'][::1,1]
        self.z_stn = self.stn_surf['vertices'][::1,2]

    def plot_cortical_projection(self):
        """Plot MNI brain including selected MNI cortical projection grid + used strip ECoG electrodes
        """

        if self.run_analysis is None:
            print("read run_analysis first")
            return

        cortex_grid = np.array(self.run_analysis.projection.grid_cortex.T)


        if self.run_analysis.settings["sess_right"] is True:
            cortex_grid[0,:] = cortex_grid[0,:]*-1
            ecog_strip = np.array(self.run_analysis.settings["coord"]["cortex_right"]["positions"]).T
        else:
            ecog_strip = np.array(self.run_analysis.settings["coord"]["cortex_left"]["positions"]).T


        fig, axes = plt.subplots(1,1, facecolor=(1,1,1), \
                                figsize=(14,9))#, dpi=300)
        axes.scatter(self.x_ecog, self.y_ecog, c="gray", s=0.001)
        axes.axes.set_aspect('equal', anchor='C')

        grid_color = self.run_analysis.projection.proj_matrix_cortex.sum(axis=1)
        pos_ecog = axes.scatter(cortex_grid[0,:],
                                cortex_grid[1,:], c=grid_color, 
                                s=30, alpha=0.8, cmap="viridis")

        pos_elec = axes.scatter(ecog_strip[0,:],
                                ecog_strip[1,:], c=np.ones(ecog_strip.shape[1]), 
                                s=50, alpha=0.8, cmap="gray", marker="x")
        plt.axis('off')
        PATH_save = os.path.join(self.feature_path, self.feature_file,
                                 "Cortical_Projection.png")
        plt.savefig(PATH_save, bbox_inches = "tight")

        print("cortical projection figure saved to: " + str(PATH_save))
