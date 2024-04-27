import numpy as np
import pandas as pd

from py_neuromodulation.nm_plots import NM_Plot


class Projection:
    def __init__(
        self,
        settings: dict,
        grid_cortex: pd.DataFrame,
        grid_subcortex: pd.DataFrame,
        coords: dict,
        nm_channels: pd.DataFrame,
        plot_projection: bool = False,
    ) -> None:

        self.test_settings(settings)

        self.grid_cortex = grid_cortex
        self.grid_subcortex = grid_subcortex
        self.coords = coords
        self.nm_channels = nm_channels
        self.project_cortex = settings["postprocessing"]["project_cortex"]
        self.project_subcortex = settings["postprocessing"]["project_subcortex"]
        self.max_dist_cortex = settings["project_cortex_settings"]["max_dist_mm"]
        self.max_dist_subcortex = settings["project_subcortex_settings"]["max_dist_mm"]
        self.ecog_channels: list # None case never handled, no need for default value
        self.lfp_channels: list # None case never handled, no need for default value

        self.idx_chs_ecog: list = []  # feature series indexes for ecog channels
        self.names_chs_ecog: list = []  # feature series name of ecog features
        self.idx_chs_lfp: list = []  # feature series indexes for lfp channels
        self.names_chs_lfp: list = []  # feature series name of lfp features
        self.feature_names: list | None = None
        self.initialized: bool = False

        self.remove_not_used_ch_from_coords()  # remove beforehand non used channels from coords

        if len(self.coords["cortex_left"]["positions"]) == 0:
            self.sess_right = True
            self.ecog_strip = self.coords["cortex_right"]["positions"]
            self.ecog_strip_names = self.coords["cortex_right"]["ch_names"]
        elif len(self.coords["cortex_right"]["positions"]) == 0:
            self.sess_right = False
            self.ecog_strip = self.coords["cortex_left"]["positions"]
            self.ecog_strip_names = self.coords["cortex_left"]["ch_names"]

        if (
            self.sess_right is True
            and len(self.coords["subcortex_right"]["positions"]) > 0
        ):
            self.lfp_elec = self.coords["subcortex_right"]["positions"]
            self.lfp_elec_names = self.coords["subcortex_right"]["ch_names"]
        elif (
            self.sess_right is False
            and len(self.coords["subcortex_left"]["positions"]) > 0
        ):
            self.lfp_elec = self.coords["subcortex_left"]["positions"]
            self.lfp_elec_names = self.coords["subcortex_left"]["ch_names"]

        self._initialize_channels()

        (
            self.proj_matrix_cortex,
            self.proj_matrix_subcortex,
        ) = self.calc_projection_matrix()

        if self.project_cortex:
            self.active_cortex_gridpoints = np.nonzero(
                self.proj_matrix_cortex.sum(axis=1)
            )[0]
        if self.project_subcortex:
            self.active_subcortex_gridpoints = np.nonzero(
                self.proj_matrix_subcortex.sum(axis=1)
            )[0]

        if plot_projection:
            nmplotter = NM_Plot(
                ecog_strip=self.ecog_strip,
                grid_cortex=self.grid_cortex.to_numpy(),
                grid_subcortex=self.grid_subcortex.to_numpy(),
                sess_right=self.sess_right,
                proj_matrix_cortex=self.proj_matrix_cortex,
            )
            nmplotter.plot_cortex()

    @staticmethod
    def test_settings(s: dict):
        if s["postprocessing"]["project_cortex"]:
            assert isinstance(
                s["project_cortex_settings"]["max_dist_mm"], (float, int)
            )
        if s["postprocessing"]["project_subcortex"]:
            assert isinstance(
                s["project_subcortex_settings"]["max_dist_mm"], (float, int)
            )

    def remove_not_used_ch_from_coords(self):
        ch_not_used = self.nm_channels.query(
            '(used==0) or (status=="bad")'
        ).name
        if len(ch_not_used) > 0:
            for ch in ch_not_used:
                for key_ in self.coords:
                    for idx, ch_coords in enumerate(
                        self.coords[key_]["ch_names"]
                    ):
                        if ch.startswith(ch_coords):
                            # delete index
                            self.coords[key_]["positions"] = np.delete(
                                self.coords[key_]["positions"], idx, axis=0
                            )
                            self.coords[key_]["ch_names"].remove(ch)

    def calc_proj_matrix(
        self,
        max_dist: float,
        grid: np.ndarray,
        coord_array: np.ndarray,
    ) -> np.ndarray:
        """Calculate projection matrix."""

        channels = coord_array.shape[0]
        distance_matrix = np.zeros([grid.shape[1], channels])

        for project_point in range(grid.shape[1]):
            for channel in range(coord_array.shape[0]):
                distance_matrix[project_point, channel] = np.linalg.norm(
                    grid[:, project_point] - coord_array[channel, :]
                )

        proj_matrix = np.zeros(distance_matrix.shape)
        for grid_point in range(distance_matrix.shape[0]):
            used_channels = np.where(distance_matrix[grid_point, :] < max_dist)[
                0
            ]

            rec_distances = distance_matrix[grid_point, used_channels]
            sum_distances : float = np.sum(1 / rec_distances)

            for _, used_channel in enumerate(used_channels):
                proj_matrix[grid_point, used_channel] = (
                    1 / distance_matrix[grid_point, used_channel]
                ) / sum_distances
        return proj_matrix

    def calc_projection_matrix(self):
        """Calculates a projection matrix based on the used coordinate arrays

        Returns
        -------
        proj_matrix_cortex (np.array)
            cortical projection_matrix in shape [grid contacts, channel contact] defaults to None
        proj_matrix_subcortex (np.array)
            subcortical projection_matrix in shape [grid contacts, channel contact] defaults to None
        """

        proj_matrix_run = np.empty(2, dtype=object)

        if self.sess_right:

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

        coord_array = [
            self.ecog_strip if grid_session[0] is not None else None,
            self.lfp_elec if grid_session[1] is not None else None,
        ]

        for loc_, grid in enumerate(grid_session):
            if loc_ == 0:  # cortex
                max_dist = self.max_dist_cortex
            elif loc_ == 1:  # subcortex
                max_dist = self.max_dist_subcortex

            if grid_session[loc_] is not None:
                proj_matrix_run[loc_] = self.calc_proj_matrix(
                    max_dist, grid, coord_array[loc_]
                )

        return proj_matrix_run[0], proj_matrix_run[1]  # cortex, subcortex

    def _initialize_channels(self) -> None:
        """Initialize channel names via nm_channel new_name column"""

        if self.project_cortex:
            self.ecog_channels = self.nm_channels.query(
                '(type=="ecog") and (used == 1) and (status=="good")'
            ).name.to_list()

            chs_ecog = self.ecog_channels.copy()
            for ecog_channel in chs_ecog:
                if ecog_channel not in self.ecog_strip_names:
                    self.ecog_channels.remove(ecog_channel)
            # write ecog_channels to be new_name
            self.ecog_channels = list(
                self.nm_channels.query("name == @self.ecog_channels").new_name
            )

        if self.project_subcortex:
            self.lfp_channels = self.nm_channels.query(
                '(type=="lfp" or type=="seeg" or type=="dbs") \
              and (used == 1) and (status=="good")'
            ).name.to_list()
            # project only channels that are in the coords
            # this also deletes channels of the other hemisphere
            chs_lfp = self.lfp_channels.copy()
            for lfp_channel in chs_lfp:
                if lfp_channel not in self.lfp_elec_names:
                    self.lfp_channels.remove(lfp_channel)
            # write lfp_channels to be new_name
            self.lfp_channels = list(
                self.nm_channels.query("name == @self.lfp_channels").new_name
            )

    def init_projection_run(self, feature_series: pd.Series) -> None:
        """Initialize indexes for respective channels in feature series computed by nm_features.py"""
        #  here it is assumed that only one hemisphere is recorded at a time!
        if self.project_cortex:
            for ecog_channel in self.ecog_channels:
                self.idx_chs_ecog.append(
                    [
                        ch_idx
                        for ch_idx, ch in enumerate(feature_series.keys())
                        if ch.startswith(ecog_channel)
                    ]
                )
                self.names_chs_ecog.append(
                    [
                        ch
                        for _, ch in enumerate(feature_series.keys())
                        if ch.startswith(ecog_channel)
                    ]
                )
            if self.names_chs_ecog:
                # get feature_names; given by ECoG sequency of features
                self.feature_names = [
                    feature_name[len(self.ecog_channels[0]) + 1 :]
                    for feature_name in self.names_chs_ecog[0]
                ]

        if self.project_subcortex:
            # for lfp_channels select here only the ones from the correct hemisphere!
            for lfp_channel in self.lfp_channels:
                self.idx_chs_lfp.append(
                    [
                        ch_idx
                        for ch_idx, ch in enumerate(feature_series.keys())
                        if ch.startswith(lfp_channel)
                    ]
                )
                self.names_chs_lfp.append(
                    [
                        ch
                        for _, ch in enumerate(feature_series.keys())
                        if ch.startswith(lfp_channel)
                    ]
                )
            if not self.feature_names and self.names_chs_lfp:
                # get feature_names; given by LFP sequency of features
                self.feature_names = [
                    feature_name[len(self.lfp_channels[0]) + 1 :]
                    for feature_name in self.names_chs_lfp[0]
                ]

        self.initialized = True

    def project_features(self, feature_series: pd.Series) -> pd.Series:
        """Project data, given idx_chs_ecog/stn"""

        if not self.initialized:
            self.init_projection_run(feature_series=feature_series)

        dat_cortex = (
            np.vstack(
                [
                    feature_series.iloc[idx_ch].values
                    for idx_ch in self.idx_chs_ecog
                ]
            )
            if self.project_cortex
            else None
        )

        dat_subcortex = (
            np.vstack(
                [
                    feature_series.iloc[idx_ch].values
                    for idx_ch in self.idx_chs_lfp
                ]
            )
            if self.project_subcortex
            else None
        )

        # project data
        proj_cortex_array: np.ndarray # get_projected_cortex_subcortex_data can return None
        proj_subcortex_array: np.ndarray # but None is not handled and will throw error in the code below
        (
            proj_cortex_array,
            proj_subcortex_array,
        ) = self.get_projected_cortex_subcortex_data(dat_cortex, dat_subcortex) # type: ignore # Ingore None return

        features_new : dict = {}
        # proj_cortex_array has shape grid_points x feature_number
        if self.project_cortex:

            features_new = features_new | {
                "gridcortex_"
                + str(act_grid_point)
                + "_"
                + feature_name: proj_cortex_array[act_grid_point, feature_idx]
                for feature_idx, feature_name in enumerate(self.feature_names) # type: ignore # Empty list handled above
                for act_grid_point in self.active_cortex_gridpoints
            }
        if self.project_subcortex:
            features_new = features_new | {
                "gridsubcortex_"
                + str(act_grid_point)
                + "_"
                + feature_name: proj_subcortex_array[
                    act_grid_point, feature_idx
                ]
                for feature_idx, feature_name in enumerate(self.feature_names) # type: ignore # Empty list handled above
                for act_grid_point in self.active_subcortex_gridpoints
            }
        feature_series = pd.concat([feature_series, pd.Series(features_new)])

        return feature_series

    def get_projected_cortex_subcortex_data(
        self,
        dat_cortex: np.ndarray | None = None,
        dat_subcortex: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
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
