"""
Lab Streaming Layer (LSL) Example
=================================

This toolbox implements the lsl ecosystem which can be utilized for offline use cases as well as live streamings
In this example the data introduced in the first demo is being analyzed
in a similar manner, This time however integrating an lsl stream.

"""

# %%
from matplotlib import pyplot as plt
import py_neuromodulation as nm

# %%
# Let’s get the example data from the provided BIDS dataset and create the channels DataFrame.

(
    RUN_NAME,
    PATH_RUN,
    PATH_BIDS,
    PATH_OUT,
    datatype,
) = nm.io.get_paths_example_data()

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm.io.read_BIDS_data(PATH_RUN=PATH_RUN)

channels = nm.utils.set_channels(
    ch_names=raw.ch_names,
    ch_types=raw.get_channel_types(),
    reference="default",
    bads=raw.info["bads"],
    new_names="default",
    used_types=("ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT"],
)

# %%
# Playing the Data
# ----------------
#
# Now we need our data to be represeted in the LSL stream.
# For this example an mne_lsl.Player is utilized, which is playing our earlier
# recorded data. However, you could make use of any LSL source (live or
# offline).
# If you want to bind your own data source, make sure to specify the
# necessary parameters (data type, type, name) accordingly.
# If you are unsure about the parameters of your data source you can
# always search for available lsl streams.
#

settings = nm.NMSettings.get_fast_compute()

player = nm.stream.LSLOfflinePlayer(
    raw=raw, stream_name="example_stream"
)

player.start_player(chunk_size=30)
# %%
# Creating the LSLStream object
# -----------------------------
#
# Next let’s create a Stream analog to the First Demo’s example However as
# we run the stream, we will set the *lsl-stream* value to True and pass
# the stream name we earlier declared when initializing the player object.

settings.features.welch = False
settings.features.fft = True
settings.features.bursts = False
settings.features.sharpwave_analysis = False
settings.features.coherence = False

# %%
stream = nm.Stream(
    sfreq=sfreq,
    channels=channels,
    settings=settings,
    coord_list=coord_list,
    verbose=True,
    line_noise=line_noise,
)
# %%
# We then simply have to set the `stream_lsl` parameter to be `True` and specify the `stream_lsl_name`.

features = stream.run(
    is_stream_lsl=True,
    plot_lsl=False,
    stream_lsl_name="example_stream",
    out_path_root=PATH_OUT,
    folder_name=RUN_NAME,
)

# %%
# We can then look at the computed features and check if the streamed data was processed correctly.
# This can be verified by the time label:

plt.plot(features.time, features.MOV_RIGHT)


######################################################################
# Feature Analysis of Movement
# ----------------------------
# We can now check the movement averaged features of an ECoG channel.
# Note that the path was here adapted to be documentation build compliant.


feature_reader = nm.analysis.FeatureReader(feature_dir=PATH_OUT, feature_file=RUN_NAME)
feature_reader.label_name = "MOV_RIGHT"
feature_reader.label = feature_reader.feature_arr["MOV_RIGHT"]

feature_reader.plot_target_averaged_channel(
    ch="ECOG_RIGHT_0",
    list_feature_keywords=None,
    epoch_len=4,
    threshold=0.5,
    ytick_labelsize=7,
    figsize_x=12,
    figsize_y=12,
)
