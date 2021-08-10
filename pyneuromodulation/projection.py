import numpy as np
from scipy import io
import os
from matplotlib import pyplot as plt
from pathlib import Path

class Projection:

    def __init__(self, settings, plot_projection=False) -> None:
        self.grid_cortex = settings["grid_cortex"]
        self.grid_subcortex = settings["grid_subcortex"]
        self.coord = settings["coord"]
        self.max_dist_cortex = settings["project_cortex_settings"]["max_dist"]
        self.max_dist_subcortex = settings["project_subcortex_settings"]["max_dist"]

        if len(self.coord["cortex_left"]["positions"]) == 0:
            self.sess_right = True
            self.ecog_strip = self.coord["cortex_right"]["positions"]
        elif len(self.coord["cortex_right"]["positions"]) == 0:
            self.sess_right = False
            self.ecog_strip = self.coord["cortex_left"]["positions"]

        self.proj_matrix_cortex, self.proj_matrix_subcortex = self.calc_projection_matrix()

        if plot_projection is True:
            self.read_plot_modules()
            self.plot_cortical_projection()

    def calc_projection_matrix(self):
        """Calculates a projection matrix based on the used coordiniate arrays
        Returns
        -------
        proj_matrix_cortex (np.array)
            cortical projection_matrix in shape [grid contacts, channel contact] defaults to None
        proj_matrix_subcortex (np.array)
            subcortical rojection_matrix in shape [grid contacts, channel contact] defaults to None
        """

        proj_matrix_run = np.empty(2, dtype=object)

        if self.sess_right is True:

            self.cortex_grid_right = np.copy(self.grid_cortex)
            self.cortex_grid_right[:, 0] = self.cortex_grid_right[:, 0] * -1

            self.subcortex_grid_right = np.copy(self.grid_subcortex)
            self.subcortex_grid_right[:, 0] = self.subcortex_grid_right[:, 0] * -1

            grid_session = [np.array(self.cortex_grid_right.T), np.array(self.subcortex_grid_right.T)]
            coord_arr = [self.ecog_strip, self.coord["subcortex_right"]["positions"]]

        else:
            grid_session = [np.array(self.grid_cortex.T), np.array(self.grid_subcortex.T)]
            coord_arr = [self.ecog_strip, self.coord["subcortex_left"]["positions"]]

        for loc_, grid in enumerate(grid_session):
            if loc_ == 0:   # cortex
                max_dist = self.max_dist_cortex
            elif loc_ == 1:  # subcortex
                max_dist = self.max_dist_subcortex

            if coord_arr[loc_] is None:  # this checks if there are cortex/subcortex channels in that run
                continue

            channels = coord_arr[loc_].shape[0]
            distance_matrix = np.zeros([grid.shape[1], channels])

            for project_point in range(grid.shape[1]):
                for channel in range(coord_arr[loc_].shape[0]):
                    distance_matrix[project_point, channel] = \
                        np.linalg.norm(grid[:, project_point] - coord_arr[loc_][channel, :])

            proj_matrix = np.zeros(distance_matrix.shape)
            for grid_point in range(distance_matrix.shape[0]):
                used_channels = np.where(distance_matrix[grid_point, :] < max_dist)[0]

                rec_distances = distance_matrix[grid_point, used_channels]
                sum_distances = np.sum(1 / rec_distances)

                for ch_idx, used_channel in enumerate(used_channels):
                    proj_matrix[grid_point, used_channel] = (1 / distance_matrix[grid_point, used_channel]) / sum_distances
            proj_matrix_run[loc_] = proj_matrix

        return proj_matrix_run[0], proj_matrix_run[1]  # cortex, subcortex

    def read_plot_modules(self,
                          PATH_PLOT=os.path.join(
                              Path(__file__).absolute().parent.parent,
                              'plots')):
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

        cortex_grid = np.array(self.grid_cortex).T
        ecog_strip = np.array(self.ecog_strip).T

        if self.sess_right is True:
            cortex_grid[0,:] = cortex_grid[0,:]*-1

        fig, axes = plt.subplots(1,1, facecolor=(1,1,1), figsize=(14,9))
        axes.scatter(self.x_ecog, self.y_ecog, c="gray", s=0.001)
        axes.axes.set_aspect('equal', anchor='C')

        grid_color = self.proj_matrix_cortex.sum(axis=1)
        pos_ecog = axes.scatter(cortex_grid[0,:],
                                cortex_grid[1,:], c=grid_color, 
                                s=30, alpha=0.8, cmap="viridis")

        pos_elec = axes.scatter(ecog_strip[0,:],
                                ecog_strip[1,:], c=np.ones(ecog_strip.shape[1]), 
                                s=50, alpha=0.8, cmap="gray", marker="x")
        plt.axis('off')

    def plot_grid_elec_3d(self):
        ax = plt.axes(projection='3d')
        #if self.sess_right is True:
        #self.cortex_grid[0,:] = self.cortex_grid[0,:]*-1
        
        # Plot becomes slow
        #ax.scatter3D(self.x_ecog[::100], self.y_ecog[::100], self.z_ecog[::100], cmap='gray', s=100, alpha=0.4)
        grid_color = self.proj_matrix_cortex.sum(axis=1)
        pos_ecog = ax.scatter3D(self.cortex_grid[0,:],
                                        self.cortex_grid[1,:], self.cortex_grid[2,:], c=grid_color, 
                                        s=300, alpha=0.8, cmap="viridis")

        pos_elec = ax.scatter(self.ecog_strip[0,:],
                                        self.ecog_strip[1,:], self.ecog_strip[2,:], c=np.ones(ecog_strip.shape[1]), 
                                        s=500, alpha=0.8, cmap="gray", marker="o")
        
    def get_projected_cortex_subcortex_data(self, dat_cortex=None, dat_subcortex=None):
        """Project cortical and subcortical data to predefined projection matrices

        Parameters
        ----------
        dat_cortex : np.ndarray, optional
            cortical features, by default None
        dat_subcortex : np.ndarray, optional
            subcortical features, by default None

        Returns
        -------
        proj_cortex : np.ndarray
            projected cortical features, by detault None
        proj_subcortex : np.ndarray
            projected subcortical features, by detault None
        """
        proj_cortex = None
        proj_subcortex = None

        if dat_cortex is not None:
            proj_cortex = self.proj_matrix_cortex @ dat_cortex
        if dat_subcortex is not None:
            proj_subcortex = self.proj_matrix_subcortex @ dat_subcortex

        return proj_cortex, proj_subcortex
