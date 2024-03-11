import numpy as np
import os
import wget

# from numba import jit
from scipy import stats
import scipy.io as sio
import pandas as pd
from typing import Union, Tuple, List
import nibabel as nib
from matplotlib import pyplot as plt

import py_neuromodulation

from py_neuromodulation import nm_plots

LIST_STRUC_UNCONNECTED_GRIDPOINTS_HULL = [256, 385, 417, 447, 819, 914]
LIST_STRUC_UNCONNECTED_GRIDPOINTS_WHOLEBRAIN = [
    1,
    8,
    16,
    33,
    34,
    35,
    36,
    37,
    51,
    75,
    77,
    78,
    99,
    109,
    115,
    136,
    155,
    170,
    210,
    215,
    243,
    352,
    359,
    361,
    415,
    416,
    422,
    529,
    567,
    569,
    622,
    623,
    625,
    627,
    632,
    633,
    634,
    635,
    639,
    640,
    641,
    643,
    644,
    650,
    661,
    663,
    667,
    683,
    684,
    685,
    704,
    708,
    722,
    839,
    840,
    905,
    993,
    1011,
]


class ConnectivityChannelSelector:

    def __init__(
        self,
        whole_brain_connectome: bool = True,
        func_connectivity: bool = True,
    ) -> None:
        """ConnectivityChannelSelector

        Parameters
        ----------
        whole_brain_connectome : bool, optional
            if True a 1236 whole-brain point grid is chosen,
            if False, a 1025 point grid of the cortical hull is loaded,
            by default True
        func_connectivity : bool, optional
            if true, functional connectivity fMRI is loaded,
            if false structural dMRIby, default True
        """

        self.connectome_name = self._get_connectome_name(
            whole_brain_connectome, func_connectivity
        )

        self.whole_brain_connectome = whole_brain_connectome
        self.func_connectivity = func_connectivity

        self.PATH_CONN_DECODING = os.path.join(
            py_neuromodulation.__path__[0],
            "ConnectivityDecoding",
        )

        if whole_brain_connectome:
            self.PATH_GRID = os.path.join(
                self.PATH_CONN_DECODING,
                "mni_coords_whole_brain.mat",
            )
            self.grid = sio.loadmat(self.PATH_GRID)["downsample_ctx"]
            if func_connectivity is False:
                # reduce the grid to only valid points that are not in LIST_STRUC_UNCONNECTED_GRIDPOINTS_WHOLEBRAIN
                self.grid = np.delete(
                    self.grid,
                    LIST_STRUC_UNCONNECTED_GRIDPOINTS_WHOLEBRAIN,
                    axis=0,
                )
        else:
            self.PATH_GRID = os.path.join(
                self.PATH_CONN_DECODING,
                "mni_coords_cortical_surface.mat",
            )
            self.grid = sio.loadmat(self.PATH_GRID)["downsample_ctx"]
            if func_connectivity is False:
                # reduce the grid to only valid points that are not in LIST_STRUC_UNCONNECTED_GRIDPOINTS_HULL
                self.grid = np.delete(
                    self.grid, LIST_STRUC_UNCONNECTED_GRIDPOINTS_HULL, axis=0
                )

        if func_connectivity:
            self.RMAP_arr = nib.load(
                os.path.join(self.PATH_CONN_DECODING, "RMAP_func_all.nii")
            ).get_fdata()
        else:
            self.RMAP_arr = nib.load(
                os.path.join(self.PATH_CONN_DECODING, "RMAP_struc.nii")
            ).get_fdata()

    def _get_connectome_name(
        self, whole_brain_connectome: str, func_connectivity: str
    ):

        connectome_name = "connectome_"
        if whole_brain_connectome:
            connectome_name += "whole_brain_"
        else:
            connectome_name += "hull_"
        if func_connectivity:
            connectome_name += "func"
        else:
            connectome_name += "struc"
        return connectome_name

    def get_available_connectomes(self) -> list:
        """Return list of saved connectomes in the
        package folder/ConnectivityDecoding/connectome_folder/ folder.

        Returns
        -------
        list_connectomes: list
        """
        return os.listdir(
            os.path.join(
                self.PATH_CONN_DECODING,
                "connectome_folder",
            )
        )

    def plot_grid(self) -> None:
        """Plot the loaded template grid that passed coordinates are matched to."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.grid[:, 0], self.grid[:, 1], self.grid[:, 2], s=50, alpha=0.2
        )
        plt.show()

    def get_closest_node(
        self, coord: Union[List, np.array]
    ) -> Tuple[List, List]:
        """Given a list or np.array of coordinates, return the closest nodes in the
        grid and their indices.

        Parameters
        ----------
        coord : np.array
            MNI coordinates with shape (num_channels, 3)

        Returns
        -------
        Tuple[List, List]
            Grid coordinates, grid indices
        """

        idx_ = []
        for c in coord:
            dist = np.linalg.norm(self.grid - c, axis=1)
            idx_.append(np.argmin(dist))

        return [self.grid[idx] for idx in idx_], idx_

    def get_rmap_correlations(
        self, fps: Union[list, np.array], RMAP_use: np.array = None
    ) -> List:
        """Calculate correlations of passed fingerprints with the RMAP

        Parameters
        ----------
        fps : Union[list, np.array]
            List of fingerprints
        RMAP_use : np.array, optional
            Passed RMAP, by default None

        Returns
        -------
        List
            correlation values
        """

        RMAP_ = self.RMAP_arr if RMAP_use is None else RMAP_use
        RMAP_ = RMAP_.flatten()
        corrs = []
        for fp in fps:
            corrs.append(np.corrcoef(RMAP_, fp.flatten())[0][1])
        return corrs

    def load_connectome(
        self,
        whole_brain_connectome: bool = None,
        func_connectivity: bool = None,
    ) -> None:
        """Load connectome, if not available download connectome from
        Zenodo.

        Parameters
        ----------
        whole_brain_connectome : bool, optional
            if true whole brain connectome
            if false cortical hull grid connectome, by default None
        func_connectivity : bool, optional
            if true fMRI if false dMRI, by default None
        """

        if whole_brain_connectome is not None:
            self.whole_brain_connectome = whole_brain_connectome
        if func_connectivity is not None:
            self.func_connectivity = func_connectivity

        self.connectome_name = self._get_connectome_name(
            self.whole_brain_connectome, self.func_connectivity
        )

        PATH_CONNECTOME = os.path.join(
            self.PATH_CONN_DECODING,
            "connectome_folder",
            self.connectome_name + ".mat",
        )

        if os.path.exists(PATH_CONNECTOME) is False:
            user_input = input(
                "Do you want to download the connectome? (yes/no): "
            ).lower()
            if user_input == "yes":
                self._download_connectome()
            elif user_input == "no":
                print("Connectome missing, has to be downloaded")

        self.connectome = sio.loadmat(PATH_CONNECTOME)

    def get_grid_fingerprints(self, grid_idx: Union[list, np.array]) -> list:
        return [self.connectome[str(grid_idx)] for grid_idx in grid_idx]

    def download_connectome(
        self,
    ):
        # download the connectome from the Zenodo API
        print("Downloading the connectome...")

        record_id = "10804702"
        file_name = self.connectome_name

        wget.download(
            f"https://zenodo.org/api/records/{record_id}/files/{file_name}/content",
            out=os.path.join(
                self.PATH_CONN_DECODING,
                "connectome_folder",
                f"{self.connectome_name}.mat",
            ),
        )


class RMAPCross_Val_ChannelSelector:

    def __init__(self) -> None:
        pass

    def load_fingerprint(self, path_nii) -> None:
        """Return Nifti fingerprint"""
        epi_img = nib.load(path_nii)
        self.affine = epi_img.affine
        fp = epi_img.get_fdata()
        return fp

    def load_all_fingerprints(
        self, path_dir: str, cond_str: str = "_AvgR_Fz.nii"
    ):

        if cond_str is not None:
            l_fps = list(filter(lambda k: cond_str in k, os.listdir(path_dir)))
        else:
            l_fps = os.listdir(path_dir)

        return l_fps, [
            self.load_fingerprint(os.path.join(path_dir, f)) for f in l_fps
        ]

    def get_fingerprints_from_path_with_cond(
        self,
        path_dir: str,
        str_to_omit: str = None,
        str_to_keep: str = None,
        keep: bool = True,
    ):

        if keep:
            l_fps = list(
                filter(
                    lambda k: "_AvgR_Fz.nii" in k and str_to_keep in k,
                    os.listdir(path_dir),
                )
            )
        else:
            l_fps = list(
                filter(
                    lambda k: "_AvgR_Fz.nii" in k and str_to_omit not in k,
                    os.listdir(path_dir),
                )
            )
        return l_fps, [
            self.load_fingerprint(os.path.join(path_dir, f)) for f in l_fps
        ]

    @staticmethod
    def save_Nii(
        fp: np.array,
        affine: np.array,
        name: str = "img.nii",
        reshape: bool = True,
    ):

        if reshape:
            fp = np.reshape(fp, (91, 109, 91), order="F")

        img = nib.nifti1.Nifti1Image(fp, affine=affine)

        nib.save(img, name)

    def get_RMAP(self, X: np.array, y: np.array):
        # faster than calculate_RMap_numba
        # https://stackoverflow.com/questions/71252740/correlating-an-array-row-wise-with-a-vector/71253141#71253141

        r = (
            len(y) * np.sum(X * y[None, :], axis=-1)
            - (np.sum(X, axis=-1) * np.sum(y))
        ) / (
            np.sqrt(
                (len(y) * np.sum(X**2, axis=-1) - np.sum(X, axis=-1) ** 2)
                * (len(y) * np.sum(y**2) - np.sum(y) ** 2)
            )
        )
        return r

    @staticmethod
    # @jit(nopython=True)
    def calculate_RMap_numba(fp, performances):
        # The RMap also needs performances; for every fingerprint / channel
        # Save the corresponding performance
        # for every voxel; correlate it with performances

        arr = fp[0].flatten()
        NUM_VOXELS = arr.shape[0]
        LEN_FPS = len(fp)
        fp_arr = np.empty((NUM_VOXELS, LEN_FPS))
        for fp_idx, fp_ in enumerate(fp):
            fp_arr[:, fp_idx] = fp_.flatten()

        RMAP = np.zeros(NUM_VOXELS)
        for voxel in range(NUM_VOXELS):
            corr_val = np.corrcoef(fp_arr[voxel, :], performances)[0][1]

            RMAP[voxel] = corr_val

        return RMAP

    @staticmethod
    # @jit(nopython=True)
    def get_corr_numba(fp, fp_test):
        val = np.corrcoef(fp_test, fp)[0][1]
        return val

    def leave_one_ch_out_cv(
        self, l_fps_names: list, l_fps_dat: list, l_per: list
    ):
        # l_fps_dat is not flattened

        per_left_out = []
        per_predict = []

        for idx_left_out, f_left_out in enumerate(l_fps_names):
            print(idx_left_out)
            l_cv = l_fps_dat.copy()
            per_cv = l_per.copy()

            l_cv.pop(idx_left_out)
            per_cv.pop(idx_left_out)

            conn_arr = []
            for f in l_cv:
                conn_arr.append(f.flatten())
            conn_arr = np.array(conn_arr)

            rmap_cv = np.nan_to_num(self.get_RMAP(conn_arr.T, np.array(per_cv)))

            per_predict.append(
                np.nan_to_num(
                    self.get_corr_numba(
                        rmap_cv, l_fps_dat[idx_left_out].flatten()
                    )
                )
            )
            per_left_out.append(l_per[idx_left_out])
        return per_left_out, per_predict

    def leave_one_sub_out_cv(
        self, l_fps_names: list, l_fps_dat: list, l_per: list, sub_list: list
    ):
        # l_fps_dat assume non flatted arrays
        # each fp including the sub_list string will be iteratively removed for test set

        per_predict = []
        per_left_out = []

        for subject_test in sub_list:
            print(subject_test)
            idx_test = [
                idx for idx, f in enumerate(l_fps_names) if subject_test in f
            ]
            idx_train = [
                idx
                for idx, f in enumerate(l_fps_names)
                if subject_test not in f
            ]
            l_cv = list(np.array(l_fps_dat)[idx_train])
            per_cv = list(np.array(l_per)[idx_train])

            conn_arr = []
            for f in l_cv:
                conn_arr.append(f.flatten())
            conn_arr = np.array(conn_arr)
            rmap_cv = np.nan_to_num(self.get_RMAP(conn_arr.T, np.array(per_cv)))

            for idx in idx_test:
                per_predict.append(
                    np.nan_to_num(
                        self.get_corr_numba(rmap_cv, l_fps_dat[idx].flatten())
                    )
                )
                per_left_out.append(l_per[idx])
        return per_left_out, per_predict

    def get_highest_corr_sub_ch(
        self,
        cohort_test: str,
        sub_test: str,
        ch_test: str,
        cohorts_train: dict,
        path_dir: str = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity",
    ):

        fp_test = self.get_fingerprints_from_path_with_cond(
            path_dir=path_dir,
            str_to_keep=f"{cohort_test}_{sub_test}_ROI_{ch_test}",
            keep=True,
        )[1][
            0
        ].flatten()  # index 1 for getting the array, 0 for the list fp that was found

        fp_pairs = []

        for cohort in cohorts_train.keys():
            for sub in cohorts_train[cohort]:
                fps_name, fps = self.get_fingerprints_from_path_with_cond(
                    path_dir=path_dir,
                    str_to_keep=f"{cohort}_{sub}_ROI",
                    keep=True,
                )

                for fp, fp_name in zip(fps, fps_name):
                    ch = fp_name[
                        fp_name.find("ROI") + 4 : fp_name.find("func") - 1
                    ]
                    corr_val = self.get_corr_numba(fp_test, fp)
                    fp_pairs.append([cohort, sub, ch, corr_val])

        idx_max = np.argmax(np.array(fp_pairs)[:, 3])
        return fp_pairs[idx_max][0:3]

    def plot_performance_prediction_correlation(
        per_left_out, per_predict, out_path_save: str = None
    ):
        df_plt_corr = pd.DataFrame()
        df_plt_corr["test_performance"] = per_left_out
        df_plt_corr["struct_conn_predict"] = (
            per_predict  # change "struct" with "funct" for functional connectivity
        )

        nm_plots.reg_plot(
            x_col="test_performance",
            y_col="struct_conn_predict",
            data=df_plt_corr,
            out_path_save=out_path_save,
        )
