"""
ECoG Movement decoding example 
==============================

This example notebook read openly accessible data from the publication 
*Electrocorticography is superior to subthalamic local field potentials 
for movement decoding in Parkinson’s disease* 
([Merk et al. 2022](https://elifesciences.org/articles/75126)). 
The dataset is available [here](https://doi.org/10.7910/DVN/IO2FLM).

For simplicity one example subject is automatically shipped within 
this repo at the *examples/data* folder, stored in 
[iEEG BIDS](https://www.nature.com/articles/s41597-019-0105-7) format.
"""

# %%
# .. note::
# In order to run this example you either have to clone the repository from the `github page <https://github.com/neuromodulation/py_neuromodulation/tree/main/examples/data>`_,
# or download the respective BIDS example subject folder, which will be referenced below.


# %%
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
    nm_stats
)
from sklearn import (
    metrics,
    model_selection,
    linear_model
)

import xgboost
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# %% 
# Let's read the example using [mne_bids](https://mne.tools/mne-bids/stable/index.html). The resulting raw object in of type [mne.RawArray](https://mne.tools/stable/generated/mne.io.RawArray.html). We can use the properties such as sampling frequency, channel names, channel types all from the mne array and create the *nm_channels* dataframe:

# replace that, the data needs to be loaded directly

RUN_NAME = "sub-000_ses-right_task-force_run-3_ieeg"

PATH_BIDS = Path.cwd() / "data"

PATH_RUN = PATH_BIDS / "sub-000" / "sess-right" / "ieeg" / (RUN_NAME + ".vhdr")

PATH_OUT = PATH_BIDS / "derivatives"

datatype = "ieeg"


(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm_IO.read_BIDS_data(
        PATH_RUN=PATH_RUN,
        BIDS_PATH=PATH_BIDS, datatype=datatype
)

nm_channels = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference="default",
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT_CLEAN","MOV_LEFT_CLEAN"]
)

nm_channels

# %%
# This example contains the force grip movement traces, we'll use the *MOV_RIGHT_CLEAN* channel as a decoding target channel. Let's check some of the raw feature and time series traces:

plt.figure(figsize=(12, 4), dpi=300)
plt.subplot(121)
plt.plot(raw.times, data[-2, :])
plt.xlabel("Time [s]")
plt.ylabel("a.u.")
plt.title("Movement label")
plt.xlim(0, 20)

plt.subplot(122)
for idx, ch_name in enumerate(nm_channels.query("used == 1").name):
    plt.plot(raw.times, data[idx, :] + idx*300, label=ch_name)
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.title("ECoG + STN-LFP time series")
plt.xlabel("Time [s]")
plt.ylabel("Voltage a.u.")
plt.xlim(0, 20)

# %%
settings = nm_settings.get_default_settings()
settings = nm_settings.set_settings_fast_compute(settings)

settings["features"]["fft"] = True
settings["features"]["bursts"] = True
settings["features"]["sharpwave_analysis"] = True
settings["features"]["coherence"] = True
settings["coherence"]["channels"] = [
    [
        "LFP_RIGHT_0",
        "ECOG_RIGHT_0"
    ]
]
settings["coherence"]["frequency_bands"] = [
    "high beta",
    "low gamma"
]
settings["sharpwave_analysis_settings"]["estimator"]["mean"] = []
for sw_feature in list(
    settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()
):
    settings["sharpwave_analysis_settings"]["sharpwave_features"][sw_feature] = True
    settings["sharpwave_analysis_settings"]["estimator"]["mean"].append(sw_feature)

# For further notebook demonstration, we will enable here alse the
# grid point projection.
settings["postprocessing"]["project_cortex"] = True

settings = nm_settings.set_settings_fast_compute(settings)  # uncomment for full documentation

# %%
stream = nm.Stream(
    sfreq=sfreq,
    nm_channels=nm_channels,
    settings=settings,
    line_noise=line_noise,
    coord_list=coord_list,
    coord_names=coord_names,
    verbose=False,
)

# %%
stream.run(
    data=data[:, :int(sfreq*60)],
    out_path_root=PATH_OUT,
    folder_name=RUN_NAME,
)

# %%
# Featue Analysis
# ---------------
# The obtained performances can now be read and visualized using the *nm_analysis.Featuer_Reader*.

# init analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME,
)
feature_reader.label_name = "MOV_LEFT_CLEAN"
feature_reader.label = feature_reader.feature_arr["MOV_LEFT_CLEAN"]

# %%
feature_reader.feature_arr

# %%
feature_reader._get_target_ch()

# %%
feature_reader.plot_target_averaged_channel(
    ch="ECOG_RIGHT_0",
    list_feature_keywords=None,
    epoch_len=4,
    threshold=0.5,
    ytick_labelsize=7,
    figsize_x=12,
    figsize_y=12
)

# %%
feature_reader.plot_all_features(
    ytick_labelsize=3,
    clim_low=-2,
    clim_high=2,
    ch_used="ECOG_RIGHT_0",
    time_limit_low_s=30,
    time_limit_high_s=60,
    normalize=True,
    save=True,
)

# %%
nm_plots.plot_corr_matrix(
    feature = feature_reader.feature_arr.filter(regex='ECOG_RIGHT_0'),
    ch_name= 'ECOG_RIGHT_0-avgref',
    feature_names=feature_reader.feature_arr.filter(regex='ECOG_RIGHT_0-avgref').columns,
    feature_file=feature_reader.feature_file,
    show_plot=True,
    figsize=(15,15),
)

# %%
# Decoding
# --------
# The main focus of the py_neuromodulation pipeline is the feature estimation. Nevertheless, the user can also use the pipeline for Machine Learning decoding. It can be used for regression and classification problems, and also using unsupervised methods, such as PCA and CCA.
# 
# Here we show an example using the XGBOOST Classifier. The labels used come from the continuous grip force movement target, namedd "MOV_LEFT_CLEAN".
# 
# First we initialize the *nm_decode.Decoder* class, which the specified *validation method*, here being a simple 3-fold cross validation, the evaluation metric, the used machine learning model, and the used channels we want to evaluate performances for.
# 
# There are are many more implemented methods, but we will here limit here to the ones presented.

model = linear_model.LinearRegression()

feature_reader.decoder = nm_decode.Decoder(
    features=feature_reader.feature_arr,
    label=feature_reader.label,
    label_name=feature_reader.label_name,
    used_chs=feature_reader.used_chs,
    model=model,
    eval_method=metrics.r2_score,
    cv_method=model_selection.KFold(n_splits=3, shuffle=True),
)

# %%
performances = feature_reader.run_ML_model(
    estimate_channels=True,
    estimate_gridpoints=False,
    estimate_all_channels_combined=True,
    save_results=True,
)

# %%
# The performances is a dictionary, that we will now transform into a dataframe:

df_per = feature_reader.get_dataframe_performances(performances)

df_per

# %%
ax = nm_plots.plot_df_subjects(
    df_per, x_col="sub", y_col="performance_test", hue="ch_type",
    PATH_SAVE= PATH_OUT / RUN_NAME / (RUN_NAME + "_decoding_performance.png")
)
ax.set_ylabel(r"$R^2$ Correlation")
ax.set_xlabel("Subject 000")
ax.set_title("Performance comparison Movement decoding")
