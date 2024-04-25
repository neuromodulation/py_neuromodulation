import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

class NiiToMNI:

    def __init__(
        self,
        PATH_nii_file: str = r"C:\code\RMap_ROI_Estimation\Automated Anatomical Labeling 3 (Rolls 2020).nii",
    ) -> None:

        self.img = nib.Nifti1Image.from_filename(PATH_nii_file)
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
        mni_coordinates: np.ndarray,
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
        ival_arr  = np.array(ival)
        coord_arr = np.array(coord_)
        ival_non_zero = ival_arr[ival != 0]
        coord_non_zero = coord_arr[ival != 0]
        print(coord_non_zero.shape)

        return coord_non_zero, ival_non_zero

    def plot_3d_coordinates(self, coord_non_zero: np.ndarray):
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


if __name__ == "__main__":

    nii_to_mni = NiiToMNI(
        PATH_nii_file=r"C:\code\py_neuromodulation\ConnectivityDecoding\Automated Anatomical Labeling 3 (Rolls 2020).nii"
    )
    mni_coordinates = nii_to_mni.downsample_nii(resampling_factor=150)
    coord_non_zero, ival_non_zero = nii_to_mni.select_non_zero_voxels(
        mni_coordinates
    )
