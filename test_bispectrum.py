"""
ECoG Movement decoding example 
==============================

"""

# %%
# This example notebook read openly accessible data from the publication
# *Electrocorticography is superior to subthalamic local field potentials
# for movement decoding in Parkinson’s disease*
# (`Merk et al. 2022 <https://elifesciences.org/articles/75126>_`).
# The dataset is available `here <https://doi.org/10.7910/DVN/IO2FLM>`_.
#
# For simplicity one example subject is automatically shipped within
# this repo at the *py_neuromodulation/data* folder, stored in
# `iEEG BIDS <https://www.nature.com/articles/s41597-019-0105-7>`_ format.

# %%
from sklearn import metrics, model_selection, linear_model
import matplotlib.pyplot as plt

import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
)

import mne

# %%
# Let's read the example using `mne_bids <https://mne.tools/mne-bids/stable/index.html>`_.
# The resulting raw object is of type `mne.RawArray <https://mne.tools/stable/generated/mne.io.RawArray.html>`_.
# We can use the properties such as sampling frequency, channel names, channel types all from the mne array and create the *nm_channels* DataFrame:
PATH_OUT = r"E:\scratch"
RUN_NAME = "TestBispectrum"

raw = mne.io.read_raw_brainvision(r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff01\ieeg\sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr")

raw.pick(["ECOG_L_1_SMC_AT", "SQUARED_ROTATION"])
sfreq = raw.info["sfreq"]

data = raw.get_data()#[:, :int(sfreq)*100]


line_noise = 50


nm_channels = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference="default",
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT"],
)

nm_channels.loc[nm_channels["name"] == "ECOG_L_1_SMC_AT", "used"] = 1
nm_channels.loc[nm_channels["name"] == "SQUARED_ROTATION", "target"] = 1

# %%
# This example contains the grip force movement traces, we'll use the *MOV_RIGHT* channel as a decoding target channel.
# Let's check some of the raw feature and time series traces:

#plt.figure(figsize=(12, 4), dpi=300)
#plt.subplot(121)
#plt.plot(raw.times, data[-1, :])
#plt.xlabel("Time [s]")
#plt.ylabel("a.u.")
#plt.title("Movement label")
#plt.xlim(0, 20)

#plt.subplot(122)
#for idx, ch_name in enumerate(nm_channels.query("used == 1").name):
#    plt.plot(raw.times, data[idx, :] + idx * 300, label=ch_name)
#plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
#plt.title("ECoG + STN-LFP time series")
#plt.xlabel("Time [s]")
#plt.ylabel("Voltage a.u.")
#plt.xlim(0, 20)

# %%
settings = nm_settings.get_default_settings()
settings = nm_settings.set_settings_fast_compute(settings)

settings["features"]["fft"] = False
settings["features"]["bursts"] = False
settings["features"]["sharpwave_analysis"] = False
settings["features"]["coherence"] = False
settings["features"]["bispectrum"] = True


# %%
stream = nm.Stream(
    sfreq=sfreq,
    nm_channels=nm_channels,
    settings=settings,
    line_noise=line_noise,
    coord_list=None,
    coord_names=None,
    verbose=True,
)

# %%
features = stream.run(
    data=data,
    out_path_root=PATH_OUT,
    folder_name=RUN_NAME,
)

# %%
# Feature Analysis Movement
# -------------------------
# The obtained performances can now be read and visualized using the :class:`nm_analysis.Feature_Reader`.

# initialize analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT,
    feature_file=RUN_NAME,
)
feature_reader.label_name = "SQUARED_ROTATION"
feature_reader.label = feature_reader.feature_arr["SQUARED_ROTATION"]

# %%
feature_reader.feature_arr.iloc[100:108, -6:]

# %%
print(feature_reader.feature_arr.shape)

# %%
feature_reader._get_target_ch()

# %%
feature_reader.plot_target_averaged_channel(
    ch="ECOG_L_1_SMC_AT",
    list_feature_keywords=None,
    epoch_len=7,
    threshold=0.5,
    ytick_labelsize=10,
    figsize_x=12,
    figsize_y=12,
)

# %%
feature_reader.plot_all_features(
    ytick_labelsize=10,
    clim_low=-2,
    clim_high=2,
    ch_used="ECOG_L_1_SMC_AT",
    time_limit_low_s=50,
    time_limit_high_s=70,
    normalize=True,
    save=True,
)

# %%
nm_plots.plot_corr_matrix(
    feature=feature_reader.feature_arr.filter(regex="ECOG_L_1_SMC_AT"),
    ch_name="ECOG_L_1_SMC_AT",
    feature_names=feature_reader.feature_arr.filter(
        regex="ECOG_L_1_SMC_AT"
    ).columns,
    feature_file=feature_reader.feature_file,
    show_plot=True,
    figsize=(15, 15),
)

# %%
# Decoding
# --------
#
# The main focus of the *py_neuromodulation* pipeline is feature estimation.
# Nevertheless, the user can also use the pipeline for machine learning decoding.
# It can be used for regression and classification problems and also dimensionality reduction such as PCA and CCA.
#
# Here, we show an example using the XGBOOST classifier. The used labels came from a continuous grip force movement target, named "MOV_RIGHT".
#
# First we initialize the :class:`~nm_decode.Decoder` class, which the specified *validation method*, here being a simple 3-fold cross validation, 
# the evaluation metric, used machine learning model, and the channels we want to evaluate performances for.
#
# There are many more implemented methods, but we will here limit it to the ones presented.

model = linear_model.LinearRegression()
feature_reader.feature_arr['SQUARED_ROTATION'] = feature_reader.feature_arr['SQUARED_ROTATION'].astype(int)>0.5

feature_reader.decoder = nm_decode.Decoder(
    features=feature_reader.feature_arr,
    label=np.array(feature_reader.label).astype(bool),
    label_name=feature_reader.label_name,
    used_chs=feature_reader.used_chs,
    model=model,
    eval_method=metrics.balanced_accuracy_score,
    cv_method=model_selection.KFold(n_splits=3, shuffle=False),
)

# %%
performances = feature_reader.run_ML_model(
    estimate_channels=True,
    estimate_gridpoints=False,
    estimate_all_channels_combined=False,
    save_results=True,
)

# %%
# The performances are a dictionary that can be transformed into a DataFrame:

df_per = feature_reader.get_dataframe_performances(performances)

df_per

# %%
ax = nm_plots.plot_df_subjects(
    df_per,
    x_col="sub",
    y_col="performance_test",
    hue="ch_type",
    PATH_SAVE=PATH_OUT / RUN_NAME / (RUN_NAME + "_decoding_performance.png"),
    figsize_tuple=(8, 5)
)
ax.set_ylabel(r"$R^2$ Correlation")
ax.set_xlabel("Subject 000")
ax.set_title("Performance comparison Movement decoding")
plt.tight_layout()
