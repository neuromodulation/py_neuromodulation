from scipy import io
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from pyneuromodulation import nm_IO

def plot_cortical_projection(
    cortex_grid: np.array,
    ecog_strip: np.array,
    grid_color: np.array,
    sess_right: bool,
    x_ecog: np.array,
    y_ecog: np.array
    ):
    """Plot MNI brain including selected MNI cortical projection grid + used strip ECoG electrodes
    Colorcoded by grid_color
    """

    if sess_right is True:
        cortex_grid[0, :] = cortex_grid[0, :]*-1

    _, axes = plt.subplots(1, 1, facecolor=(1, 1, 1), figsize=(14, 9))
    axes.scatter(x_ecog, y_ecog, c="gray", s=0.001)
    axes.axes.set_aspect('equal', anchor='C')

    
    _ = axes.scatter(cortex_grid[:, 0],
                            cortex_grid[:, 1], c=grid_color,
                            s=30, alpha=0.8, cmap="viridis")

    _ = axes.scatter(ecog_strip[:, 0],
                            ecog_strip[:, 1], c=np.ones(ecog_strip.shape[0]),
                            s=50, alpha=0.8, cmap="gray", marker="x")
    plt.axis('off')

def plot_grid_elec_3d(
    cortex_grid: np.array,
    ecog_strip: np.array,
    grid_color: np.array
    ):
    
    ax = plt.axes(projection='3d')
    # if self.sess_right is True:
    # self.cortex_grid[0,:] = self.cortex_grid[0,:]*-1

    # Plot becomes slow
    # ax.scatter3D(self.x_ecog[::100], self.y_ecog[::100], self.z_ecog[::100], cmap='gray', s=100, alpha=0.4)

    _ = ax.scatter3D(cortex_grid[:, 0],
                            cortex_grid[:, 1], cortex_grid[:, 2], c=grid_color,
                            s=300, alpha=0.8, cmap="viridis")

    _ = ax.scatter(ecog_strip[:, 0],
                            ecog_strip[:, 1], ecog_strip[:, 2], c=np.ones(ecog_strip.shape[0]),
                            s=500, alpha=0.8, cmap="gray", marker="o")

class NM_Plot():

    def __init__(self, ecog_strip: np.array = None,
        grid_cortex: np.array = None,
        grid_subcortex: np.array = None,
        sess_right: bool = False,
        proj_matrix_cortex: np.array = None
        ) -> None:

        self.grid_cortex = grid_cortex
        self.grid_subcortex = grid_subcortex
        self.ecog_strip = ecog_strip
        self.sess_right = sess_right
        self.proj_matrix_cortex = proj_matrix_cortex

        if proj_matrix_cortex is not None:
            self.grid_color = self.proj_matrix_cortex.sum(axis=1)

        self.faces, self.vertices, self.grid, self.stn_surf, self.x_ver, self.y_ver, \
            self.x_ecog, self.y_ecog, self.z_ecog, \
            self.x_stn, self.y_stn, self.z_stn = nm_IO.read_plot_modules()

    def plot_cortical_projection(self):
        """Plot grid plots color coded respectively of a projection matrix"""

        if self.proj_matrix_cortex is not None:
            plot_cortical_projection(np.array(self.grid_cortex),
                np.array(self.ecog_strip), self.grid_color,
                self.sess_right, self.x_ecog, self.y_ecog)
        else:
            raise ValueError("no projection matrix supplied")

    def plot_grid_elec_3d(self):
        if self.proj_matrix_cortex is not None:
            plot_grid_elec_3d(np.array(self.cortex_grid),
                np.array(self.ecog_strip), self.grid_color)
        else:
            raise ValueError("no projection matrix supplied")