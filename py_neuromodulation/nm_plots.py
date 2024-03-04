from scipy import stats
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from typing import Optional
import seaborn as sb
import pandas as pd
import logging

logger = logging.getLogger("PynmLogger")

from py_neuromodulation import nm_IO, nm_stats


def plot_df_subjects(
    df,
    x_col="sub",
    y_col="performance_test",
    hue=None,
    title="channel specific performances",
    PATH_SAVE: str = None,
    figsize_tuple: tuple = (5, 3),
):
    alpha_box = 0.4
    plt.figure(figsize=figsize_tuple, dpi=300)
    sb.boxplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        showmeans=False,
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        whiskerprops={"linewidth": 2, "zorder": 10, "alpha": alpha_box},
        capprops={"alpha": alpha_box},
        medianprops=dict(
            linestyle="-", linewidth=5, color="gray", alpha=alpha_box
        ),
    )

    ax = sb.stripplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        dodge=True,
        s=5,
    )

    if hue is not None:
        n_hues = df[hue].nunique()

        handles, labels = ax.get_legend_handles_labels()
        l = plt.legend(
            handles[0:n_hues],
            labels[0:n_hues],
            bbox_to_anchor=(1.05, 1),
            loc=2,
            title=hue,
            borderaxespad=0.0,
        )
    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    if PATH_SAVE is not None:
        plt.savefig(
            PATH_SAVE,
            bbox_inches="tight",
        )
    # plt.show()
    return plt.gca()


def plot_epoch(
    X_epoch: np.array,
    y_epoch: np.array,
    feature_names: list,
    z_score: bool = None,
    epoch_len: int = 4,
    sfreq: int = 10,
    str_title: str = None,
    str_label: str = None,
    ytick_labelsize: float = None,
):
    if z_score is None:
        X_epoch = stats.zscore(
            np.nan_to_num(np.nanmean(np.squeeze(X_epoch), axis=0)),
            axis=0,
            nan_policy="omit",
        ).T
    y_epoch = np.stack(np.array(y_epoch))
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.imshow(X_epoch, aspect="auto")
    plt.yticks(
        np.arange(0, len(feature_names), 1), feature_names, size=ytick_labelsize
    )
    plt.xticks(
        np.arange(0, X_epoch.shape[1], 1),
        np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2),
        rotation=90,
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Time [s]")
    plt.title(str_title)

    plt.subplot(212)
    for i in range(y_epoch.shape[0]):
        plt.plot(y_epoch[i, :], color="black", alpha=0.4)
    plt.plot(
        y_epoch.mean(axis=0),
        color="black",
        alpha=1,
        linewidth=3.0,
        label="mean target",
    )
    plt.legend()
    plt.ylabel("Target")
    plt.title(str_label)
    plt.xticks(
        np.arange(0, X_epoch.shape[1], 1),
        np.round(np.arange(-epoch_len / 2, epoch_len / 2, 1 / sfreq), 2),
        rotation=90,
    )
    plt.xlabel("Time [s]")
    plt.tight_layout()


def reg_plot(
    x_col: str, y_col: str, data: pd.DataFrame, out_path_save: str = None
):
    plt.figure(figsize=(4, 4), dpi=300)
    rho, p = nm_stats.permutationTestSpearmansRho(
        data[x_col],
        data[y_col],
        False,
        "R^2",
        5000,
    )
    sb.regplot(x=x_col, y=y_col, data=data)
    plt.title(f"{y_col}~{x_col} p={np.round(p, 2)} rho={np.round(rho, 2)}")

    if out_path_save is not None:
        plt.savefig(
            out_path_save,
            bbox_inches="tight",
        )


def plot_bar_performance_per_channel(
    ch_names,
    performances: dict,
    PATH_OUT: str,
    sub: str = None,
    save_str: str = "ch_comp_bar_plt.png",
    performance_metric: str = "Balanced Accuracy",
):
    """
    performances dict is output of ml_decode
    """
    plt.figure(figsize=(4, 3), dpi=300)
    if sub is None:
        sub = list(performances.keys())[0]
    plt.bar(
        np.arange(len(ch_names)),
        [performances[sub][p]["performance_test"] for p in performances[sub]],
    )
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.xlabel("channels")
    plt.ylabel(performance_metric)
    plt.savefig(
        os.path.join(PATH_OUT, save_str),
        bbox_inches="tight",
    )
    plt.close()


def plot_corr_matrix(
    feature: pd.DataFrame,
    feature_file: str = None,
    ch_name: str = None,
    feature_names: list[str] = None,
    show_plot=True,
    OUT_PATH: str = None,
    feature_name_plt="Features_corr_matr",
    save_plot: bool = False,
    save_plot_name: str = None,
    figsize: tuple[int] = (7, 7),
    title: str = None,
    cbar_vmin: float = -1,
    cbar_vmax: float = 1.0,
):
    # cut out channel name for each column
    if ch_name is not None:
        feature_col_name = [
            i[len(ch_name) + 1 :] for i in feature_names if ch_name in i
        ]
    else:
        feature_col_name = feature.columns

    plt.figure(figsize=figsize)
    if feature_names is not None:
        corr = feature[feature_names].corr()
    else:
        corr = feature.corr()
    sb.heatmap(
        corr,
        xticklabels=feature_col_name,
        yticklabels=feature_col_name,
        vmin=cbar_vmin,
        vmax=cbar_vmax,
        cmap="viridis",
    )
    if title is None:
        if ch_name is not None:
            plt.title("Correlation matrix features channel: " + str(ch_name))
        else:
            plt.title("Correlation matrix")
    else:
        plt.title(title)

    # if len(feature_col_name) > 50:
    #    plt.xticks([])
    #    plt.yticks([])

    if save_plot and save_plot_name is None:
        plt_path = get_plt_path(
            OUT_PATH=OUT_PATH,
            feature_file=feature_file,
            ch_name=ch_name,
            str_plt_type=feature_name_plt,
            # feature_name=feature_names.__str__,  # This here raises an error in os.path.join in line 251
        )
    if save_plot and save_plot_name is not None:
        plt_path = os.path.join(OUT_PATH, save_plot_name)

    if save_plot:
        plt.savefig(plt_path, bbox_inches="tight")
        logger.info(f"Correlation matrix figure saved to {plt_path}")

    if show_plot is False:
        plt.close()

    plt.tight_layout()

    return plt.gca()


def plot_feature_series_time(features) -> None:
    plt.imshow(features.T, aspect="auto")


def get_plt_path(
    OUT_PATH: str | None = None,
    feature_file: str | None = None,
    ch_name: str | None = None,
    str_plt_type: str | None = None,
    feature_name: str | None = None,
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
                feature_file,
                str_plt_type + "_ch_" + ch_name + ".png",
            )
        else:
            plt_path = os.path.join(
                OUT_PATH,
                feature_file,
                str_plt_type + "_ch_" + ch_name + "_" + feature_name + ".png",
            )
    elif None not in (OUT_PATH, feature_file) and ch_name is None:
        plt_path = os.path.join(
            OUT_PATH,
            feature_file,
            str_plt_type + "_ch_" + feature_name + ".png",
        )

    else:
        plt_path = os.getcwd() + ".png"
    return plt_path


def plot_epochs_avg(
    X_epoch: np.ndarray,
    y_epoch: np.ndarray,
    epoch_len: int,
    sfreq: int,
    feature_names: list[str] = None,
    feature_str_add: str = None,
    cut_ch_name_cols: bool = True,
    ch_name: str = None,
    label_name: str = None,
    normalize_data: bool = True,
    show_plot: bool = True,
    save: bool = False,
    OUT_PATH: str = None,
    feature_file: str = None,
    str_title: str = "Movement aligned features",
    ytick_labelsize=None,
    figsize_x: float = 8,
    figsize_y: float = 8,
) -> None:
    # cut channel name of for axis + "_" for more dense plot
    if feature_names is None:
        if cut_ch_name_cols and None not in (ch_name, feature_names):
            feature_names = [
                i[len(ch_name) + 1 :]
                for i in list(feature_names)
                if ch_name in i
            ]

    if normalize_data:
        X_epoch_mean = stats.zscore(
            np.nanmean(np.squeeze(X_epoch), axis=0), axis=0, nan_policy="omit"
        ).T
    else:
        X_epoch_mean = np.nanmean(np.squeeze(X_epoch), axis=0).T

    if len(X_epoch_mean.shape) == 1:
        X_epoch_mean = np.expand_dims(X_epoch_mean, axis=0)

    plt.figure(figsize=(figsize_x, figsize_y))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
    plt.subplot(gs[0])
    plt.imshow(X_epoch_mean, aspect="auto")
    plt.yticks(
        np.arange(0, len(feature_names), 1), feature_names, size=ytick_labelsize
    )
    plt.xticks(
        np.arange(0, X_epoch.shape[1], int(X_epoch.shape[1] / 10)),
        np.round(np.arange(-epoch_len / 2, epoch_len / 2, epoch_len / 10), 2),
        rotation=90,
    )
    plt.xlabel("Time [s]")
    str_title = str_title
    if ch_name:
        str_title += f" channel: {ch_name}"
    plt.title(str_title)

    plt.subplot(gs[1])
    for i in range(y_epoch.shape[0]):
        plt.plot(y_epoch[i, :], color="black", alpha=0.4)
    plt.plot(
        y_epoch.mean(axis=0),
        color="black",
        alpha=1,
        linewidth=3.0,
        label="mean target",
    )
    plt.legend()
    plt.ylabel("Target")
    plt.title(label_name)
    plt.xticks(
        np.arange(0, X_epoch.shape[1], int(X_epoch.shape[1] / 10)),
        np.round(np.arange(-epoch_len / 2, epoch_len / 2, epoch_len / 10), 2),
        rotation=90,
    )
    plt.xlabel("Time [s]")
    plt.tight_layout()

    if save:
        plt_path = get_plt_path(
            OUT_PATH,
            feature_file,
            ch_name,
            str_plt_type="MOV_aligned_features",
            feature_name=feature_str_add,
        )
        plt.savefig(plt_path, bbox_inches="tight")
        logger.info(f"Feature epoch average figure saved to: {str(plt_path)}")
    if show_plot is False:
        plt.close()


def plot_grid_elec_3d(
    cortex_grid: np.ndarray | None = None,
    ecog_strip: np.ndarray | None = None,
    grid_color: np.ndarray | None = None,
    strip_color: np.ndarray | None = None,
):
    ax = plt.axes(projection="3d")

    if cortex_grid is not None:
        grid_color = (
            np.ones(cortex_grid.shape[0]) if grid_color is None else grid_color
        )
        _ = ax.scatter3D(
            cortex_grid[:, 0],
            cortex_grid[:, 1],
            cortex_grid[:, 2],
            c=grid_color,
            s=300,
            alpha=0.8,
            cmap="viridis",
        )

    if ecog_strip is not None:
        strip_color = (
            np.ones(ecog_strip.shape[0]) if strip_color is None else strip_color
        )
        _ = ax.scatter(
            ecog_strip[:, 0],
            ecog_strip[:, 1],
            ecog_strip[:, 2],
            c=strip_color,
            s=500,
            alpha=0.8,
            cmap="gray",
            marker="o",
        )


def plot_all_features(
    df: pd.DataFrame,
    time_limit_low_s: float = None,
    time_limit_high_s: float = None,
    normalize: bool = True,
    ytick_labelsize: int = 4,
    clim_low: float = None,
    clim_high: float = None,
    save: bool = False,
    title="all_feature_plt.pdf",
    OUT_PATH: str = None,
    feature_file: str = None,
):
    if time_limit_high_s is not None:
        df = df[df["time"] < time_limit_high_s * 1000]
    if time_limit_low_s is not None:
        df = df[df["time"] > time_limit_low_s * 1000]

    cols_plt = [c for c in df.columns if c != "time"]
    if normalize is True:
        data_plt = stats.zscore(df[cols_plt], nan_policy="omit")
    else:
        data_plt = df[cols_plt]

    plt.figure()  # figsize=(7, 5), dpi=300
    plt.imshow(data_plt.T, aspect="auto")
    plt.xlabel("Time [s]")
    plt.ylabel("Feature Names")
    plt.yticks(np.arange(len(cols_plt)), cols_plt, size=ytick_labelsize)

    tick_num = np.arange(0, df.shape[0], int(df.shape[0] / 10))
    tick_labels = np.array(np.rint(df["time"].iloc[tick_num] / 1000), dtype=int)
    plt.xticks(tick_num, tick_labels)

    plt.title(f"Feature Plot {feature_file}")

    if clim_low is not None:
        plt.clim(vmin=clim_low)
    if clim_high is not None:
        plt.clim(vmax=clim_high)

    plt.colorbar()
    plt.tight_layout()

    if save is True:
        plt_path = os.path.join(OUT_PATH, feature_file, title)
        plt.savefig(plt_path, bbox_inches="tight")


class NM_Plot:
    def __init__(
        self,
        ecog_strip: np.ndarray | None = None,
        grid_cortex: np.ndarray | None = None,
        grid_subcortex: np.ndarray | None = None,
        sess_right: Optional[bool] = False,
        proj_matrix_cortex: np.ndarray | None = None,
    ) -> None:
        self.grid_cortex = grid_cortex
        self.grid_subcortex = grid_subcortex
        self.ecog_strip = ecog_strip
        self.sess_right = sess_right
        self.proj_matrix_cortex = proj_matrix_cortex

        (
            self.faces,
            self.vertices,
            self.grid,
            self.stn_surf,
            self.x_ver,
            self.y_ver,
            self.x_ecog,
            self.y_ecog,
            self.z_ecog,
            self.x_stn,
            self.y_stn,
            self.z_stn,
        ) = nm_IO.read_plot_modules()

    def plot_grid_elec_3d(self) -> None:
        plot_grid_elec_3d(np.array(self.grid_cortex), np.array(self.ecog_strip))

    def plot_cortex(
        self,
        grid_cortex: Optional[np.ndarray] = None,
        grid_color: Optional[np.ndarray] = None,
        ecog_strip: Optional[np.ndarray] = None,
        strip_color: Optional[np.ndarray] = None,
        sess_right: Optional[bool] = None,
        save: bool = False,
        OUT_PATH: str = None,
        feature_file: str = None,
        feature_str_add: str = None,
        show_plot: bool = True,
        title: str = "Cortical grid",
        set_clim: bool = True,
        lower_clim: float = 0.5,
        upper_clim: float = 0.7,
        cbar_label: str = "Balanced Accuracy",
    ):
        """Plot MNI brain including selected MNI cortical projection grid + used strip ECoG electrodes
        Colorcoded by grid_color
        """

        if grid_cortex is None:
            if type(self.grid_cortex) is pd.DataFrame:
                grid_cortex = np.array(self.grid_cortex)
            else:
                grid_cortex = self.grid_cortex

        if ecog_strip is None:
            ecog_strip = self.ecog_strip

        if sess_right is True:
            grid_cortex[0, :] = grid_cortex[0, :] * -1

        fig, axes = plt.subplots(1, 1, facecolor=(1, 1, 1), figsize=(14, 9))
        axes.scatter(self.x_ecog, self.y_ecog, c="gray", s=0.01)
        axes.axes.set_aspect("equal", anchor="C")

        if grid_cortex is not None:
            grid_color = (
                np.ones(grid_cortex.shape[0])
                if grid_color is None
                else grid_color
            )

            pos_ecog = axes.scatter(
                grid_cortex[:, 0],
                grid_cortex[:, 1],
                c=grid_color,
                s=150,
                alpha=0.8,
                cmap="viridis",
                label="grid points",
            )
            if set_clim:
                pos_ecog.set_clim(lower_clim, upper_clim)
        if ecog_strip is not None:
            strip_color = (
                np.ones(ecog_strip.shape[0])
                if strip_color is None
                else strip_color
            )

            pos_ecog = axes.scatter(
                ecog_strip[:, 0],
                ecog_strip[:, 1],
                c=strip_color,
                s=400,
                alpha=0.8,
                cmap="viridis",
                marker="x",
                label="ecog electrode",
            )
        plt.axis("off")
        plt.legend()
        plt.title(title)
        if set_clim:
            pos_ecog.set_clim(lower_clim, upper_clim)
            cbar = fig.colorbar(pos_ecog)
            cbar.set_label(cbar_label)

        if save:
            plt_path = get_plt_path(
                OUT_PATH,
                feature_file,
                ch_name=None,
                str_plt_type="PLOT_CORTEX",
                feature_name=feature_str_add,
            )
            plt.savefig(plt_path, bbox_inches="tight")
            logger.info(
                f"Feature epoch average figure saved to: {str(plt_path)}"
            )
        if show_plot is False:
            plt.close()
