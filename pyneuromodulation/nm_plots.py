from scipy import stats
import os
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
import seaborn as sns
import pandas as pd

from pyneuromodulation import nm_IO

def plot_corr_matrix(
    feature: pd.DataFrame,
    feature_file,
    ch_name: str=None,
    feature_names: list[str]=None,
    show_plot=False,
    OUT_PATH:str = None,
    feature_name_plt="Features_corr_matr",
    save_plot:bool = True
    ):

    # cut out channel name for each column
    feature_col_name = [i[len(ch_name)+1:] for i in feature_names if ch_name in i]

    plt.figure(figsize=(7, 7))
    corr = feature.corr()
    sns.heatmap(corr,
                xticklabels=feature_col_name,
                yticklabels=feature_col_name)
    plt.title("Features channel: " + str(ch_name))

    if save_plot:
        plt_path = get_plt_path(
            OUT_PATH=OUT_PATH,
            feature_file=feature_file,
            ch_name=ch_name,
            str_plt_type=feature_name_plt,
            feature_name=feature_names.__str__
        )

        plt.savefig(plt_path, bbox_inches="tight")
        print("Correlation matrix figure saved to " + str(plt_path))

    if show_plot is False:
        plt.close()

def get_plt_path(
    OUT_PATH: str = None,
    feature_file: str = None,
    ch_name:str = None,
    str_plt_type:str = None,
    feature_name:str = None
    ) -> None:
    """[summary]

    Parameters
    ----------
    OUT_PATH : str, optional
        folder of preprocessed runs, by default None
    feature_file : str, optional
        run_name, by default None
    ch_name : str, optional
        ch_name, by default None
    str_plt_type : str, optional
        type of plot, e.g. mov_avg_feature or corr_matr, by default None
    feature_name : str, optional
        e.g. bandpower, stft, sharpwave_prominence, by default None
    """
    if None not in (ch_name, OUT_PATH, feature_file):
        if feature_name is None:
            plt_path = os.path.join(
                OUT_PATH,
                feature_file[:-len(".vhdr")],
                str_plt_type + "_ch_" + ch_name + ".png"
            )
        else:
            plt_path = os.path.join(
                OUT_PATH,
                feature_file[:-len(".vhdr")],
                str_plt_type + "_ch_" \
                     + ch_name + "_" + feature_name + ".png"
            )
    elif None not in (OUT_PATH, feature_file) and ch_name is None:
        plt_path = os.path.join(
                OUT_PATH,
                feature_file[:-len(".vhdr")],
                str_plt_type + "_ch_" \
                     + feature_name + ".png"
            )

    else:
        plt_path = os.getcwd() + ".png"
    return plt_path
    

def plot_epochs_avg(
    X_epoch: np.array,
    y_epoch: np.array,
    epoch_len: int,
    sfreq: int,
    feature_names:list[str]=None,
    feature_str_add: str=None,
    cut_ch_name_cols:bool = True,
    ch_name:str = None,
    label_name: str = None,
    normalize_data:bool = True,
    show_plot:bool = True,
    save:bool = False,
    OUT_PATH: str = None,
    feature_file: str=None
    ):

    # cut channel name of for axis + "_" for more dense plot
    if cut_ch_name_cols and None not in (ch_name, feature_names):
        feature_names = [i[len(ch_name)+1:] for i in list(feature_names) if ch_name in i]

    if normalize_data:
        X_epoch_mean = stats.zscore(np.mean(np.squeeze(X_epoch), axis=0), axis=0).T
    else:
        X_epoch_mean = np.mean(np.squeeze(X_epoch), axis=0).T

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.imshow(X_epoch_mean, aspect='auto')
    plt.yticks(np.arange(0, len(feature_names), 1), feature_names)
    plt.xticks(np.arange(0, X_epoch.shape[1], 1),
                np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2), rotation=90)
    plt.xlabel("Time [s]")
    str_title = "Movement aligned features"
    if ch_name:
        str_title += f" channel: {ch_name}"
    plt.title(str_title)

    plt.subplot(212)
    for i in range(y_epoch.shape[0]):
        plt.plot(y_epoch[i, :], color="black", alpha=0.4)
    plt.plot(y_epoch.mean(axis=0), color="black", alpha=1, linewidth=3.0, label="mean target")
    plt.legend()
    plt.ylabel("target")
    plt.title(label_name)
    plt.xticks(np.arange(0, X_epoch.shape[1], 1),
                np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2), rotation=90)
    plt.xlabel("Time [s]")
    plt.tight_layout()

    if save:
        plt_path = get_plt_path(
            OUT_PATH,
            feature_file,
            ch_name,
            str_plt_type="MOV_aligned_features",
            feature_name=feature_str_add
        )
        plt.savefig(plt_path, bbox_inches="tight")
        print("Feature epoch average figure saved to: " + str(plt_path))
    if show_plot is False:
        plt.close()

def plot_grid_elec_3d(
    cortex_grid: Optional[np.array]=None,
    ecog_strip: Optional[np.array]=None,
    grid_color: Optional[np.array]=None,
    strip_color: Optional[np.array] = None,
    ):

    ax = plt.axes(projection='3d')

    if cortex_grid is not None:
        grid_color = np.ones(cortex_grid.shape[0]) if grid_color is None else grid_color
        _ = ax.scatter3D(cortex_grid[:, 0],
                            cortex_grid[:, 1], cortex_grid[:, 2], c=grid_color,
                            s=300, alpha=0.8, cmap="viridis")
    
    if ecog_strip is not None:
        strip_color = np.ones(ecog_strip.shape[0]) if strip_color is None else strip_color
        _ = ax.scatter(ecog_strip[:, 0],
                            ecog_strip[:, 1], ecog_strip[:, 2], c=strip_color,
                            s=500, alpha=0.8, cmap="gray", marker="o")

class NM_Plot():

    def __init__(self,
        ecog_strip: Optional[np.array] = None,
        grid_cortex: Optional[np.array] = None,
        grid_subcortex: Optional[np.array] = None,
        sess_right: Optional[bool] = False,
        proj_matrix_cortex: Optional[np.array] = None
        ) -> None:

        self.grid_cortex = grid_cortex
        self.grid_subcortex = grid_subcortex
        self.ecog_strip = ecog_strip
        self.sess_right = sess_right
        self.proj_matrix_cortex = proj_matrix_cortex

        self.faces, self.vertices, self.grid, self.stn_surf, self.x_ver, self.y_ver, \
            self.x_ecog, self.y_ecog, self.z_ecog, \
            self.x_stn, self.y_stn, self.z_stn = nm_IO.read_plot_modules()

    def plot_grid_elec_3d(self):
        
        plot_grid_elec_3d(
            np.array(self.cortex_grid),
            np.array(self.ecog_strip)
        )

    def plot_cortex(
        self,
        cortex_grid: Optional[np.array] = None,
        grid_color: Optional[np.array] = None,
        ecog_strip: Optional[np.array] = None,
        strip_color: Optional[np.array] = None,
        sess_right: Optional[bool] = None,
        save: bool = False,
        OUT_PATH: str = None,
        feature_file: str=None,
        feature_str_add : str=None,
        show_plot:bool=True,
        set_clim:bool=True
        ):
        """Plot MNI brain including selected MNI cortical projection grid + used strip ECoG electrodes
        Colorcoded by grid_color
        """

        if cortex_grid is None:
            cortex_grid = self.grid_cortex

        if ecog_strip is None:
            ecog_strip = self.ecog_strip

        if sess_right is True:
            cortex_grid[0, :] = cortex_grid[0, :]*-1

        fig, axes = plt.subplots(1, 1, facecolor=(1, 1, 1), figsize=(14, 9))
        axes.scatter(self.x_ecog, self.y_ecog, c="gray", s=0.01)
        axes.axes.set_aspect('equal', anchor='C')

        if cortex_grid is not None:

            grid_color = np.ones(cortex_grid.shape[0]) if grid_color is None else grid_color

            pos_ecog = axes.scatter(cortex_grid[:, 0],
                                    cortex_grid[:, 1], c=grid_color,
                                    s=150, alpha=0.8, cmap="viridis")
        if ecog_strip is not None:
            strip_color = np.ones(ecog_strip.shape[0]) if strip_color is None else strip_color

            pos_ecog = axes.scatter(ecog_strip[:, 0],
                                ecog_strip[:, 1], c=strip_color,
                                s=400, alpha=0.8, cmap="viridis", marker="x")
        plt.axis('off')
        if set_clim:
            pos_ecog.set_clim(0.5, 0.8)
            cbar = fig.colorbar(pos_ecog)
            cbar.set_label("Balanced Accuracy")

        if save:
            plt_path = get_plt_path(
                OUT_PATH,
                feature_file,
                ch_name=None,
                str_plt_type="PLOT_CORTEX",
                feature_name=feature_str_add
            )
            plt.savefig(plt_path, bbox_inches="tight")
            print("Feature epoch average figure saved to: " + str(plt_path))
        if show_plot is False:
            plt.close()
