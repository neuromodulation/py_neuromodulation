# Run here the feature estimation steps in the Berlin data set subject, then export it to example_decodingML
# Take sub 006, ses MedOff02, SelfpacedRotation L, Stim ON.
import os
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
)

sub = "008"
ses = "EcogLfpMedOff01"
task = "SelfpacedRotationR"
acq = "StimOn"
run = 1
datatype = "ieeg"

# Define run name and access paths in the BIDS format.
RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_run-{run}"

PATH_RUN = os.path.join(
    "/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data",
    f"sub-{sub}",
    f"ses-{ses}",
    datatype,
    RUN_NAME,
)
PATH_BIDS = "/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data"

# Provide a path for the output data.
PATH_OUT = os.path.join("/home/lauraflyra/Documents/BCCN/Lab_Rotation_DBS_Decoding/BIDS_data",
                        "derivatives")

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
    used_types=("ecog", "dbs"),
    target_keywords=("ANALOG_R_ROTA_CH", ),  # This defines which channel is gonna be
    # used as target later during the decoding steps
)

stream = nm.Stream(
    settings=None,
    nm_channels=nm_channels,
    path_grids=None,
    verbose=True,
)

stream.reset_settings()

# We first take care of the preprocessing steps - and here we want to perform all of them in the order given by stream.settings['preprocessing']['preprocessing_order']

stream.settings['preprocessing']['raw_resampling'] = True
stream.settings['preprocessing']['raw_normalization'] = True
stream.settings['preprocessing']['re_referencing'] = True
stream.settings['preprocessing']['notch_filter'] = True
stream.settings['preprocessing']['preprocessing_order'] = [
            "raw_resampling",
            "notch_filter",
            "re_referencing",
            "raw_normalization"
        ]

# Now we focus on the features that we want to estimate:

stream.settings['features']['raw_hjorth'] = True
stream.settings['features']['bandpass_filter'] = True
stream.settings['features']['fft'] = True
stream.settings['features']['sharpwave_analysis'] = True
stream.settings['features']['fooof'] = False
stream.settings['features']['nolds'] = False

# Then we set the postprocessing steps
stream.settings['postprocessing']['feature_normalization'] = False
stream.settings['postprocessing']['project_cortex'] = True
stream.settings['postprocessing']['project_subcortex'] = True

stream.init_stream(
    sfreq=sfreq,
    line_noise=line_noise,
    coord_list=coord_list,
    coord_names=coord_names,
)

stream.run(
    data=data,
    out_path_root=PATH_OUT,
    folder_name=RUN_NAME,
)


