"""
Grid Point Projection
=====================

"""

# %%
# In ECoG datasets the electrode locations are usually different. For this reason, we established a grid
# with a set of points defined in a standardized MNI brain.
# Data is then interpolated to this grid, such that they are common across patients, which allows across patient decoding use cases.
# 
# In this notebook, we will plot these grid points and see how the features extracted from our data can be projected into this grid space.
# 
# In order to do so, we'll read saved features that were computed in the ECoG movement notebook.
# Please note that in order to do so, when running the feature estimation, the settings
# 
# .. note::
#
#     .. code-block:: python
# 
#         stream.settings['postprocessing']['project_cortex'] = True
#         stream.settings['postprocessing']['project_subcortex'] = True
# 
#     need to be set to `True` for a cortical and/or subcortical projection.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt

import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_plots,
    nm_IO,
    nm_settings,
    nm_define_nmchannels
)


# %%
# Read features from BIDS data
# ----------------------------
#
# We first estimate features, with the `grid_point` projection settings enabled for cortex. 


# %%
RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT, datatype = nm_IO.get_paths_example_data()

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm_IO.read_BIDS_data(
        PATH_RUN=PATH_RUN
)

settings = nm_settings.get_default_settings()
settings = nm_settings.set_settings_fast_compute(settings)

settings["postprocessing"]["project_cortex"] = True

nm_channels = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference="default",
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT_CLEAN","MOV_LEFT_CLEAN"]
)

stream = nm.Stream(
    sfreq=sfreq,
    nm_channels=nm_channels,
    settings=settings,
    line_noise=line_noise,
    coord_list=coord_list,
    coord_names=coord_names,
    verbose=True,
)

features = stream.run(
    data=data[:, :int(sfreq*5)],
    out_path_root=PATH_OUT,
    folder_name=RUN_NAME,
)

# %%
# From nm_analysis.py, we use the :class:~`nm_analysis.FeatureReader` class to load the data.

# init analyzer
feature_reader = nm_analysis.FeatureReader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME
)

# %%
# To perform the grid projection, for all computed features we check for every grid point if there is any electrode channel within the spatial range ```max_dist_mm```, and weight 
# this electrode contact by the inverse distance and normalize across all electrode distances within the maximum distance range.
# This gives us a projection matrix that we can apply to streamed data, to transform the feature-channel matrix *(n_features, n_channels)* into the grid point matrix *(n_features, n_gridpoints)*.
# 
# To save computation time, this projection matrix is precomputed before the real time run computation. 
# The cortical grid is stored in *py_neuromodulation/grid_cortex.tsv* and the electrodes coordinates are stored in *_space-mni_electrodes.tsv* in a BIDS dataset.
# 
# .. note::
#
#     One remark is that our cortical and subcortical grids are defined for the **left** hemisphere of the brain and, therefore, electrode contacts are mapped to the left hemisphere.
# 
# From the analyzer, the user can plot the cortical projection with the function below, display the grid points and ECoG electrodes are crosses.
# The yellow grid points are the ones that are active for that specific ECoG electrode location. The inactive grid points are shown in purple.

feature_reader.plot_cort_projection()

# %%
# We can also plot only the ECoG electrodes or the grid points, with the help of the data saved in feature_reader.sidecar. BIDS sidecar files are json files where you store additional information, here it is used to save the ECoG strip positions and the grid coordinates, which are not part of the settings and nm_channels.csv. We can check what is stored in the file and then use the nmplotter.plot_cortex function:

grid_plotter = nm_plots.NM_Plot(
    ecog_strip=np.array(feature_reader.sidecar["coords"]["cortex_right"]["positions"]),
    grid_cortex=np.array(feature_reader.sidecar["grid_cortex"]),
    # grid_subcortex=np.array(feature_reader.sidecar["grid_subcortex"]),
    sess_right=feature_reader.sidecar["sess_right"],
    proj_matrix_cortex=np.array(feature_reader.sidecar["proj_matrix_cortex"])
)

# %%
grid_plotter.plot_cortex(
    grid_color=np.sum(np.array(feature_reader.sidecar["proj_matrix_cortex"]),axis=1),
    lower_clim=0.,
    upper_clim=1.0,
    cbar_label="Used Grid Points",
    title = "ECoG electrodes projected onto cortical grid"
)

# %%
feature_reader.sidecar["coords"]["cortex_right"]["positions"]

# %%
feature_reader.nmplotter.plot_cortex(
    ecog_strip=np.array(
        feature_reader.sidecar["coords"]["cortex_right"]["positions"],
    ),
    lower_clim=0.,
    upper_clim=1.0,
    cbar_label="Used ECoG Electrodes",
    title = "Plot of ECoG electrodes"
)

# %%
feature_reader.nmplotter.plot_cortex(
    np.array(
        feature_reader.sidecar["grid_cortex"]
    ),
    lower_clim=0.,
    upper_clim=1.0,
    cbar_label="All Grid Points",
    title = "All grid points"
)

# %%
# The Projection Matrix
# ---------------------
# To go from the feature-channel matrix *(n_features, n_channels)* to the grid point matrix *(n_features, n_gridpoints)*
# we need a projection matrix that has the shape *(n_channels, n_gridpoints)*.
# It maps the strengths of the signals in each ECoG channel to the correspondent ones in the cortical grid.
# In the cell below we plot this matrix, that has the property that the column sum over channels for each grid point is either 1 or 0.

plt.figure(figsize=(8,5))
plt.imshow(np.array(feature_reader.sidecar['proj_matrix_cortex']), aspect = 'auto')
plt.colorbar(label = "Strength of ECoG signal in each grid point")
plt.xlabel("ECoG channels")
plt.ylabel("Grid points")
plt.title("Matrix mapping from ECoG to grid")

# %%
# Feature Plot in the Grid: An Example of Post-processing
# -------------------------------------------------------
# First we take the dataframe with all the features in all time points.

df = feature_reader.feature_arr

# %%
df.iloc[:5, :5]

# %%
# Then we filter for only 'avgref_fft_theta', which gives us the value for fft_theta in all 6 ECoG channels over all time points. Then we take only the 6th time point - as an arbitrary choice.

fft_theta_oneTimePoint = np.asarray(df[df.columns[df.columns.str.contains(pat = 'avgref_fft_theta')]].iloc[5])
fft_theta_oneTimePoint

# %%
# Then the projection of the features into the grid is gonna be the color of the grid points in the *plot_cortex* function.
# That is the matrix multiplication of the projection matrix of the cortex and 6 values for the *fft_theta* feature above.

grid_fft_Theta = np.array(feature_reader.sidecar["proj_matrix_cortex"]) @ fft_theta_oneTimePoint

feature_reader.nmplotter.plot_cortex(np.array(
    feature_reader.sidecar["grid_cortex"]),grid_color = grid_fft_Theta, set_clim = True, lower_clim=min(grid_fft_Theta[grid_fft_Theta>0]), upper_clim=max(grid_fft_Theta), cbar_label="FFT Theta Projection to Grid", title = "FFT Theta Projection to Grid")

# %%
# Lower and upper boundaries for clim were chosen to be the max and min values of the projection of the features (minimum value excluding zero). This can be checked in the cell below:

grid_fft_Theta

# %%
# In the plot above we can see how the intensity of the fast fourier transform in the theta band varies for each grid point in the cortex, for one specific time point.
