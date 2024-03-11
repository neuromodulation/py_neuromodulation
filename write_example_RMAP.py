from py_neuromodulation import nm_RMAP

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    ch_sel = nm_RMAP.ConnectivityChannelSelector(
        whole_brain_connectome=True, func_connectivity=False
    )

    ch_sel.get_available_connectomes()

    ch_sel.load_connectome()
    # ch_sel.plot_grid()

    mni_coords = [[10, 40, 20], [50, 14, 12]]
    grid_coords, grid_idxs = ch_sel.get_closest_node(mni_coords)

    grid_fps = ch_sel.get_grid_fingerprints(grid_idxs)

    corrs = ch_sel.get_rmap_correlations(grid_fps, ch_sel.RMAP_arr)
