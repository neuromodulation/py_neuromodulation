"""
Lab Streaming Layer (LSL) Example
=================================

This toolbox implemnts the lsl ecosystem which can be utilized for offline use cases as well as live streamings
---------------------------------------------------------------------------------------------------------------

In this example the data introduced in the first demo is being analyzed
in a similar manner, This time however integrating a lsl stream.

"""
# %%
from matplotlib import pyplot as plt
from py_neuromodulation import (nm_mnelsl_generator, nm_IO, nm_define_nmchannels, nm_analysis, nm_stream_offline, nm_settings, nm_generator)

# %%
######################################################################
# Let’s get some data and create the nm_channels DataFrame
# 

(
    RUN_NAME,
    PATH_RUN,
    PATH_BIDS,
    PATH_OUT,
    datatype,
) = nm_IO.get_paths_example_data()

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm_IO.read_BIDS_data(
    PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
)

nm_channels = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference="default",
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT"],
)

nm_channels
print(nm_IO.get_paths_example_data())

# %%
plt.figure(figsize=(12, 4), dpi=300)
plt.subplot(121)
plt.plot(raw.times, data[-1, :])
plt.xlabel("Time [s]")
plt.ylabel("a.u.")
plt.title("Movement label")
plt.xlim(0, 20)

plt.subplot(122)
for idx, ch_name in enumerate(nm_channels.query("used == 1").name):
    plt.plot(raw.times, data[idx, :] + idx * 300, label=ch_name)
plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
plt.title("ECoG + STN-LFP time series")
plt.xlabel("Time [s]")
plt.ylabel("Voltage a.u.")
plt.xlim(0, 20)

# %%
######################################################################
# Playing the Data
# ~~~~~~~~~~~~~~~~
# 
# | Now we need our Data to be produced in some way.
# | For this example a LSL Player is utilized which is playing our earlier
#   recorderd data. However, you could make use of any LSL source (live or
#   offline).
# | If you want to bind your own data source, make sure to specify the
#   necessariy parameters (data type, type, name) accordingly.
# | If you are unsure about the parameters of your data source you can
#   always search for available lsl streams.
# 

settings = nm_settings.get_default_settings()
settings = nm_settings.set_settings_fast_compute(settings)

player = nm_mnelsl_generator.LSLOfflinePlayer(data= data, sfreq = 1000, stream_name="example_stream")
player.start_player()
# %%
######################################################################
# Creating the LSLSTream Object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Next let’s create a Stream analog to the First Demo’s example However as
# we run the stream, we will set the *lsl-stream* value to True and pass
# the stream name we earlier declared when initializing the player object
# 
# %%
settings["features"]["welch"] = True
settings["features"]["fft"] = True
settings["features"]["bursts"] = True
settings["features"]["sharpwave_analysis"] = True
settings["features"]["coherence"] = True
settings["coherence"]["channels"] = [["LFP_RIGHT_0", "ECOG_RIGHT_0"]]
settings["coherence"]["frequency_bands"] = ["high beta", "low gamma"]
settings["sharpwave_analysis_settings"]["estimator"]["mean"] = []
for sw_feature in list(
    settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()
):
    settings["sharpwave_analysis_settings"]["sharpwave_features"][
        sw_feature
    ] = True
    settings["sharpwave_analysis_settings"]["estimator"]["mean"].append(
        sw_feature
    )
# %%
stream = nm_stream_offline.Stream(
    sfreq=sfreq, nm_channels=nm_channels, settings=settings, coord_list=coord_list,verbose=True, line_noise=line_noise,
)
# %%
features = stream.run(stream_lsl= True, stream_lsl_name="example_stream")
# %%
features.head()


######################################################################
# Feature Analysis of Movement
# ----------------------------
# 
# %%
feature_reader = nm_analysis.Feature_Reader(feature_dir ="./",feature_file = "./sub")
feature_reader.label_name = "MOV_RIGHT"
feature_reader.label = feature_reader.feature_arr["MOV_RIGHT"]
feature_reader.feature_arr.iloc[100:108, -6:]

# %%
print(feature_reader.feature_arr.shape)
print(feature_reader._get_target_ch())


# %%
feature_reader.plot_target_averaged_channel(
    ch="ECOG_RIGHT_0",
    list_feature_keywords=None,
    epoch_len=4,
    threshold=0.5,
    ytick_labelsize=7,
    figsize_x=12,
    figsize_y=12,
)

