import nibabel as nib
import numpy as np
import scipy.io as sio
import os


def write_connectome_mat(
    PATH_Fingerprints: str = r"D:\Connectome_RMAP_OUT\ROIs\HCP1000 6K",
    PATH_CONNECTOME: str = os.path.join(
        "py_neuromodulation",
        "ConnectivityDecoding",
        "connectome_struct.mat",
    ),
    func_: bool = False,
):

    # connectome = sio.loadmat(PATH_CONNECTOME)  # check if read was successful

    # load all fingerprints and put them in .npy
    dict_connectome = {}
    if func_ is False:
        files_fps = [f for f in os.listdir(PATH_Fingerprints) if ".nii" in f]
    else:
        files_fps = [
            f
            for f in os.listdir(PATH_Fingerprints)
            if "func_seed_AvgR_Fz.nii" in f
        ]

    # I except 1025 files, check which ones are missing
    missing_files = []

    for i in range(1, 1026):

        MISSING = False

        if func_ is False:
            if f"ROI-{i}_struc_seed.nii" not in files_fps:
                missing_files.append(f"ROI-{i}_struc_seed.nii")
                MISSING = True
        else:
            if f"ROI-{i}_func_seed_AvgR_Fz.nii" not in files_fps:
                missing_files.append(f"ROI-{i}_func_seed_AvgR_Fz.nii")
                MISSING = True

        if MISSING:
            ROI_file = os.path.join(
                r"D:\Connectome_RMAP_OUT\whole_brain\ROIs", f"ROI-{i}.nii"
            )
            # copy the ROI file to the following folder:
            PATH_ROI_OUT = (
                r"D:\Connectome_RMAP_OUT\whole_brain\ROI_missing_struc"
            )
            import shutil

            shutil.copy(ROI_file, os.path.join(PATH_ROI_OUT, f"ROI-{i}.nii"))

    for idx, f in enumerate(files_fps):
        # load the .nii file and put it all in in a dictionary with the name of the file
        fp = (
            nib.load(os.path.join(PATH_Fingerprints, f))
            .get_fdata()
            .astype(np.float16)
        )
        if "struc" in f:
            dict_connectome[f[f.find("ROI-") + 4 : f.find("_struc")]] = fp
        else:
            dict_connectome[
                f[f.find("ROI-") + 4 : f.find("_func_seed_AvgR_Fz.nii")]
            ] = fp

        print(idx)
    # save the dictionary
    sio.savemat(
        PATH_CONNECTOME,
        dict_connectome,
    )


if __name__ == "__main__":

    write_connectome_mat(
        PATH_Fingerprints=r"D:\Connectome_RMAP_OUT\whole_brain\struc\HCP1000 6K",
        PATH_CONNECTOME=os.path.join(
            "py_neuromodulation",
            "ConnectivityDecoding",
            "connectome_whole_brain_struc.mat",
        ),
    )  # 58 files are missing

    write_connectome_mat(
        PATH_Fingerprints=r"D:\Connectome_RMAP_OUT\whole_brain\func",
        PATH_CONNECTOME=os.path.join(
            "py_neuromodulation",
            "ConnectivityDecoding",
            "connectome_whole_brain_func.mat",
        ),
        func_=True,
    )

    write_connectome_mat(
        PATH_Fingerprints=r"D:\Connectome_RMAP_OUT\hull\func\GSP 1000 (Yeo 2011)_Full Set (Yeo 2011)",
        PATH_CONNECTOME=os.path.join(
            "py_neuromodulation",
            "ConnectivityDecoding",
            "connectome_hull_func.mat",
        ),
        func_=True,
    )  # all there

    write_connectome_mat(
        PATH_Fingerprints=r"D:\Connectome_RMAP_OUT\hull\struc\HCP1000 6K",
        PATH_CONNECTOME=os.path.join(
            "py_neuromodulation",
            "ConnectivityDecoding",
            "connectome_hull_struc.mat",
        ),
    )  # 5 missing
