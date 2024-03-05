import enum
import nibabel as nib
import numpy as np
import os

# from numba import jit
from scipy import stats
import pandas as pd

from py_neuromodulation import nm_plots


class RMAPChannelSelector:

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
            # print(idx_left_out)
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
            # print(subject_test)
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
