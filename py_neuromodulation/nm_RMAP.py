import enum
import nibabel as nib
import numpy as np
import os
from numba import jit

class RMAPChannelSelector:
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_fingerprint(path_nii) -> None:
        """Return flattened Nifti fingerprint"""
        epi_img = nib.load(path_nii)
        fp = epi_img.get_fdata()
        return fp

    def get_fingerprints_from_path_with_cond(
        self,
        path_dir: str,
        str_to_omit : str = None,
        str_to_keep : str = None,
        keep: bool = True,
    ):

        if keep:
            l_fps = list(
                filter(
                    lambda k: '_AvgR_Fz.nii' in k and str_to_keep in k,
                        os.listdir(path_dir)
                )
            )
        else:
            l_fps = list(
                filter(
                    lambda k: '_AvgR_Fz.nii' in k and str_to_omit not in k,
                        os.listdir(path_dir)
                )
            )
        return l_fps, [
            self.load_fingerprint(
                os.path.join(path_dir, f)
             ) for f in l_fps
        ]

    def save_Nii(fp, name="img.nii", reshape=True):
        if reshape:
            fp = np.reshape((91, 109, 91), order="F")

        xform = np.eye(4) * 2
        img = nib.nifti1.Nifti1Image(fp, xform)

        nib.save(img, name)

    @staticmethod
    @jit(nopython=True)
    def calculate_RMap(fp, performances):
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
            RMAP[voxel] = np.corrcoef(
                    fp_arr[voxel, :],
                    performances
                )[0][1]

        return RMAP

    @staticmethod
    @jit(nopython=True)
    def get_corr_numba(fp_test, fp):
        fp_flattened = fp.flatten()
        val = np.corrcoef(fp_test, fp_flattened)[0,1]
        return val

    def get_highest_corr_sub_ch(
        self,
        cohort_test : str,
        sub_test : str,
        ch_test : str,
        cohorts_train : dict,
        path_dir : str = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity"
    ):

        fp_test = self.get_fingerprints_from_path_with_cond(
            path_dir=path_dir,
            str_to_keep=f"{cohort_test}_{sub_test}_ROI_{ch_test}",
            keep=True
        )[1][0].flatten()  # index 1 for getting the array, 0 for the list fp that was found

        fp_pairs = []

        for cohort in cohorts_train.keys():
            for sub in cohorts_train[cohort]:
                fps_name, fps = self.get_fingerprints_from_path_with_cond(
                    path_dir=path_dir,
                    str_to_keep=f"{cohort}_{sub}_ROI",
                    keep=True
                )

                for fp, fp_name in zip(fps, fps_name):
                    ch = fp_name[fp_name.find("ROI")+4:fp_name.find("func")-1]
                    corr_val = self.get_corr_numba(fp_test, fp)
                    fp_pairs.append(
                        [
                            cohort,
                            sub,
                            ch,
                            corr_val
                        ]
                    )

        idx_max = np.argmax(np.array(fp_pairs)[:,3])
        return fp_pairs[idx_max][0:3]
