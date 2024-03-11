import nibabel as nib
import numpy as np
import scipy.io as sio
import os
from matplotlib import pyplot as plt


class NiiToMNI:

    def __init__(
        self,
        PATH_nii_file: str = r"C:\code\RMap_ROI_Estimation\Automated Anatomical Labeling 3 (Rolls 2020).nii",
    ) -> None:

        self.img = nib.load(PATH_nii_file)
        self.data = self.img.get_fdata()

    def downsample_nii(
        self,
        resampling_factor: int = 150,
    ):

        # PATH_MNI_TO_ATLAS = r"C:\code\mni_to_atlas\src\mni_to_atlas\atlases\AAL.nii"
        # img_mni_to_atlas = nib.load(PATH_MNI_TO_ATLAS)

        x_dim, y_dim, z_dim = self.data.shape

        # Create arrays of voxel coordinates
        x_coords, y_coords, z_coords = np.meshgrid(
            range(x_dim), range(y_dim), range(z_dim), indexing="ij"
        )

        # Downsample here the voxels --> check lateron if the voxels have non-zero values
        x_c_flatten = x_coords.flatten()[::resampling_factor]
        y_c_flatten = y_coords.flatten()[::resampling_factor]
        z_c_flatten = z_coords.flatten()[::resampling_factor]

        # Combine coordinates into a single array
        voxel_coordinates = np.column_stack(
            (
                x_c_flatten,
                y_c_flatten,
                z_c_flatten,
                np.ones(x_c_flatten.shape[0]),
            )
        )

        aff_m = self.img.affine
        aff_m[0, 0] = 2
        aff_m[0, 3] = -90

        mni_coordinates = np.dot(aff_m, voxel_coordinates.T).T[:, :3]

        return mni_coordinates

    def select_non_zero_voxels(
        self,
        mni_coordinates: np.array,
    ):

        coords = np.hstack(
            (mni_coordinates, np.ones((mni_coordinates.shape[0], 1)))
        )

        # and transform back to get the voxel values
        voxels_downsampled = np.array(
            np.linalg.solve(self.img.affine, coords.T).T
        ).astype(int)[:, :3]

        ival = []
        coord_ = []
        for i in range(voxels_downsampled.shape[0]):
            ival.append(self.data[tuple(voxels_downsampled[i, :])])
            coord_.append(mni_coordinates[i, :])

        # get only voxel values non-zero
        ival = np.array(ival)
        coord_ = np.array(coord_)
        ival_non_zero = ival[ival != 0]
        coord_non_zero = coord_[ival != 0]
        print(coord_non_zero.shape)

        return coord_non_zero, ival_non_zero


def write_connectome_mat(
    PATH_Fingerprints: str = r"D:\Connectome_RMAP_OUT\ROIs\HCP1000 6K",
    PATH_CONNECTOME: str = os.path.join(
        "py_neuromodulation",
        "ConnectivityDecoding",
        "connectome_struct.mat",
    ),
):

    # connectome = sio.loadmat(PATH_CONNECTOME)  # check if read was successful

    # load all fingerprints and put them in .npy
    dict_connectome = {}
    files_fps = [f for f in os.listdir(PATH_Fingerprints) if ".nii" in f]
    for f in files_fps:
        # load the .nii file and put it all in in a dictionary with the name of the file
        fp = (
            nib.load(os.path.join(PATH_Fingerprints, f))
            .get_fdata()
            .astype(np.float16)
        )
        dict_connectome[f[f.find("ROI-") + 4 : f.find("_struc")]] = fp

    # save the dictionary
    sio.savemat(
        PATH_CONNECTOME,
        dict_connectome,
    )


if __name__ == "__main__":

    nii_to_mni = NiiToMNI(
        PATH_nii_file=r"C:\code\RMap_ROI_Estimation\Automated Anatomical Labeling 3 (Rolls 2020).nii"
    )
    mni_coordinates = nii_to_mni.downsample_nii(resampling_factor=150)
    coord_non_zero, ival_non_zero = nii_to_mni.select_non_zero_voxels(
        mni_coordinates
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coord_non_zero[:, 0],
        coord_non_zero[:, 1],
        coord_non_zero[:, 2],
        s=50,
        alpha=0.2,
    )
    plt.show()
