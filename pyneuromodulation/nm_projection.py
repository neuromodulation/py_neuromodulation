import numpy as np
import pandas as pd
from scipy import io
import os
from matplotlib import pyplot as plt
from pathlib import Path

from pyneuromodulation import nm_plots

class Projection:

    def __init__(self, settings: dict,
                grid_cortex: pd.DataFrame,
                grid_subcortex: pd.DataFrame,
                coord: dict,
                plot_projection: bool=False,
                ) -> None:

        self.grid_cortex = grid_cortex
        self.grid_subcortex = grid_subcortex
        self.coord = coord
        self.max_dist_cortex = settings["project_cortex_settings"]["max_dist"]
        self.max_dist_subcortex = settings["project_subcortex_settings"]["max_dist"]

        if len(self.coord["cortex_left"]["positions"]) == 0:
            self.sess_right = True
            self.ecog_strip = self.coord["cortex_right"]["positions"]
        elif len(self.coord["cortex_right"]["positions"]) == 0:
            self.sess_right = False
            self.ecog_strip = self.coord["cortex_left"]["positions"]

        self.project_cortex = settings["methods"]["project_cortex"]
        self.project_subcortex = settings["methods"]["project_subcortex"]

        self.proj_matrix_cortex, self.proj_matrix_subcortex = self.calc_projection_matrix()

        if plot_projection is True:
            nmplotter = nm_plots.NM_Plot(self.ecog_strip, self.grid_cortex, self.grid_subcortex,
                self.sess_right, self.proj_matrix_cortex)
            nmplotter.plot_cortical_projection()

    def calc_proj_matrix(self, max_dist, grid: np.array, coord_array: np.array):

        channels = coord_array.shape[0]
        distance_matrix = np.zeros([grid.shape[1], channels])

        for project_point in range(grid.shape[1]):
            for channel in range(coord_array.shape[0]):
                distance_matrix[project_point, channel] = \
                    np.linalg.norm(grid[:, project_point] - coord_array[channel, :])

        proj_matrix = np.zeros(distance_matrix.shape)
        for grid_point in range(distance_matrix.shape[0]):
            used_channels = np.where(distance_matrix[grid_point, :] < max_dist)[0]

            rec_distances = distance_matrix[grid_point, used_channels]
            sum_distances = np.sum(1 / rec_distances)

            for _, used_channel in enumerate(used_channels):
                proj_matrix[grid_point, used_channel] = \
                    (1 / distance_matrix[grid_point, used_channel]) / sum_distances
        return proj_matrix

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

            if self.project_cortex:
                cortex_grid_right = np.copy(self.grid_cortex)
                cortex_grid_right[:, 0] = cortex_grid_right[:, 0] * -1
                self.cortex_grid_right = np.array(cortex_grid_right.T)
            else:
                self.cortex_grid_right = None

            if self.project_subcortex:
                subcortex_grid_right = np.copy(self.grid_subcortex)
                subcortex_grid_right[:, 0] = subcortex_grid_right[:, 0] * -1
                self.subcortex_grid_right = np.array(subcortex_grid_right).T
            else:
                self.subcortex_grid_right = None

            grid_session = [self.cortex_grid_right, self.subcortex_grid_right]
            coord_array = [self.ecog_strip, self.coord["subcortex_right"]["positions"]]

        else:
            if self.project_cortex:
                self.cortex_grid_left = np.array(self.grid_cortex.T)
            else:
                self.cortex_grid_left = None
            if self.project_subcortex:
                self.subcortex_grid_left = np.array(self.grid_subcortex.T)
            else:
                self.subcortex_grid_left = None

            grid_session = [self.cortex_grid_left, self.subcortex_grid_left]
            coord_array = [self.ecog_strip, self.coord["subcortex_left"]["positions"]]

        for loc_, grid in enumerate(grid_session):
            if loc_ == 0:   # cortex
                max_dist = self.max_dist_cortex
            elif loc_ == 1:  # subcortex
                max_dist = self.max_dist_subcortex

            if grid_session[loc_] is not None:
                proj_matrix_run[loc_] = self.calc_proj_matrix(max_dist, grid, coord_array[loc_])

        return proj_matrix_run[0], proj_matrix_run[1]  # cortex, subcortex

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
