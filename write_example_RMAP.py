from py_neuromodulation import nm_RMAP

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    ch_sel = nm_RMAP.ConnectivityChannelSelector()

    # make a 3D plot of ch_sel.grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        ch_sel.grid[:, 0], ch_sel.grid[:, 1], ch_sel.grid[:, 2], s=50, alpha=0.2
    )
    plt.show()

    mni_coords = [[10, 40, 20], [50, 14, 12]]
    fps, grid_idxs = ch_sel.get_closest_node(mni_coords)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        np.array(mni_coords)[0, :],
        np.array(mni_coords)[1, :],
        np.array(mni_coords)[2, :],
        s=50,
        alpha=0.2,
    )
    plt.show()

    corrs = ch_sel.get_rmap_correlations(fps, struct_corr=True)
