import os
from re import VERBOSE

import numpy as np
import scipy.io as spio
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import pickle

from py_neuromodulation import nm_EpochStream, nm_GenericStream

from yaml import load


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def set_settings(settings: dict):
    for method in list(settings["methods"].keys()):
        settings["methods"][method] = False

    settings["methods"]["fft"] = True
    settings["methods"]["fooof"] = True
    settings["methods"]["return_raw"] = True
    settings["methods"]["raw_hjorth"] = True
    settings["methods"]["re_referencing"] = False
    settings["methods"]["raw_normalization"] = False
    settings["methods"]["feature_normalization"] = True

    settings["fooof"]["periodic"]["center_frequency"] = False
    settings["fooof"]["periodic"]["band_width"] = False
    settings["fooof"]["periodic"]["height_over_ap"] = False

    settings["methods"]["sharpwave_analysis"] = True

    for key in list(
        settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()
    ):
        settings["sharpwave_analysis_settings"]["sharpwave_features"][
            key
        ] = True
    settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "peak_left"
    ] = False
    settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "peak_right"
    ] = False
    settings["sharpwave_analysis_settings"][
        "apply_estimator_between_peaks_and_troughs"
    ] = True
    settings["sharpwave_analysis_settings"]["filter_low_cutoff"] = 5
    settings["sharpwave_analysis_settings"]["filter_high_cutoff"] = 40
    return settings


def run_patient_GenericStream(f):

    file_name = os.path.basename(f)[: -len("_edit.mat")]
    dat = loadmat(os.path.join(PATH_DATA, f))["D"]
    labels = dat["labels"][~np.array(dat["bad"], dtype="bool")]

    data = np.swapaxes(np.swapaxes(dat["data"], 0, 2), 1, 2)
    data = data[~np.array(dat["bad"], dtype="bool"), :, :]
    NUM_CH = data.shape[1]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    label_arr = enc.fit_transform(integer_encoded.reshape(-1, 1))

    label_arr_concat = np.concatenate(
        (np.expand_dims(integer_encoded + 1, axis=1), label_arr), axis=1
    )  # integer encoded + 1, since REST will be set to zero

    label_arr_epochs = np.zeros([data.shape[0], 4, data.shape[2]])
    label_arr_epochs_names = list(label_encoder.classes_)
    label_arr_epochs_names.insert(0, "ALL")

    integer_encoded_names = label_arr_epochs_names.copy()
    integer_encoded_names[0] = "REST"

    # label_arr_epochs columns: ALL, NTR, PLS, UNPLS
    # integer_encoded_names: 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS

    arr_insert = np.repeat(label_arr_concat[:, :, np.newaxis], 1000, axis=2)
    label_arr_epochs[:, :, 3500:4500] = arr_insert

    data_comb = np.concatenate((data, label_arr_epochs), axis=1)
    data_stream = np.concatenate(data_comb, axis=1)

    ch_names = list(dat["ch_names"])
    ch_names = ch_names + label_arr_epochs_names

    ch_types = ["seeg" for _ in range(NUM_CH)]
    ch_types = ch_types + ["misc" for _ in range(len(label_arr_epochs_names))]

    stream = nm_GenericStream.GenericStream(
        READ_BIDS=False,
        ch_names=ch_names,
        ch_types=ch_types,
        fs=dat["fsample"],
        line_noise=50,
        PATH_NM_CHANNELS=None,
        bads=[],
        reference=None,
        PATH_OUT=os.path.join(PATH_DATA, "features_epochs_nonorm"),
        VERBOSE=False,
    )

    stream.set_data(data_stream)
    stream.settings = set_settings(stream.settings)
    stream.nm_channels.loc[
        stream.nm_channels.query('type == "misc"').index, "target"
    ] = 1

    stream.run()

    stream.save_after_stream(file_name)


def run_patient_EpochStream(f):
    epoch_stream = nm_EpochStream.EpochStream(VERBOSE=False)
    # epoch_stream.read_epoch_data(
    #    os.path.join("scripts", "preproc_ecog_v0.3_Jan22_rerefSegments_data.npy")
    # )
    file_name = os.path.basename(f)[: -len("_edit.mat")]

    dat = loadmat(os.path.join(PATH_DATA, f))["D"]
    labels = dat["labels"][~np.array(dat["bad"], dtype="bool")]
    # bring data into shape (samples, channels, time)
    epoch_stream.data = np.swapaxes(np.swapaxes(dat["data"], 0, 2), 1, 2)

    # use only good trials
    epoch_stream.data = epoch_stream.data[
        ~np.array(dat["bad"], dtype="bool"), :, :
    ]

    epoch_stream.set_fs(dat["fsample"])
    epoch_stream.set_linenoise(50)
    NUM_CH = epoch_stream.data.shape[1]

    epoch_stream.nm_channels = epoch_stream._get_nm_channels(
        PATH_NM_CHANNELS=None,
        ch_names=dat["ch_names"],
        ch_types=["lfp" for _ in range(NUM_CH)],
        bads=[],
        used_types=["lfp" for _ in range(NUM_CH)],
        target_keywords=[],
        reference=None,
    )

    epoch_stream.settings = set_settings(epoch_stream.settings)
    epoch_stream.run()

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    label_arr = enc.fit_transform(integer_encoded.reshape(-1, 1))

    dict_out = {}
    dict_out["label_" + label_encoder.classes_[0]] = label_arr[:, 0]
    dict_out["label_" + label_encoder.classes_[1]] = label_arr[:, 1]
    dict_out["label_" + label_encoder.classes_[2]] = label_arr[:, 2]

    dict_out["label"] = labels
    dict_out["feature_names"] = list(epoch_stream.feature_arr.columns)

    # add labels to nm_channels
    for label_name in ["label_PLS", "label_UNPLS", "label_NTR"]:
        epoch_stream.nm_channels = epoch_stream.nm_channels.append(
            {
                "name": label_name,
                "rereference": None,
                "used": 0,
                "target": 1,
                "type": "misc",
                "status": "good",
                "new_name": label_name,
            },
            ignore_index=True,
        )

    epoch_stream.path_out = os.path.join(PATH_DATA, "features_epochs_nonorm")

    dict_out["features"] = np.stack(
        [
            np.array(epoch_stream.feature_arr_list[idx])
            for idx in range(len(epoch_stream.feature_arr_list))
        ]
    )

    with open(
        os.path.join(
            PATH_DATA, "features_epochs_nonorm", file_name, "dict_out.p"
        ),
        "wb",
    ) as fp:
        pickle.dump(dict_out, fp)

    epoch_stream.save_after_stream(file_name, save_features=False)

    print(f"finished processing {file_name}")


PATH_DATA = r"C:\Users\ICN_admin\Documents\TRD Analysis"


def main():

    files = [f for f in os.listdir(PATH_DATA) if "_edit" in f]
    # for f in files:
    #    run_patient_GenericStream(f)
    pool = Pool(processes=len(files))
    pool.map(run_patient_GenericStream, files)


if __name__ == "__main__":
    main()
