import mne
import mne_bids
import pybv  # pip install pybv
from pathlib import PurePath


def set_chtypes(vhdr_raw):
    """
    define MNE RawArray channel types
    """
    print("Setting new channel types...")
    remapping_dict = {}
    for ch_name in vhdr_raw.info["ch_names"]:
        if ch_name.startswith("ECOG"):
            remapping_dict[ch_name] = "ecog"
        elif ch_name.startswith(("LFP", "STN")):
            remapping_dict[ch_name] = "dbs"
        elif ch_name.startswith("EMG"):
            remapping_dict[ch_name] = "emg"
        elif ch_name.startswith("EEG"):
            remapping_dict[ch_name] = "misc"
        elif ch_name.startswith(("MOV", "ANALOG", "ROT", "ACC", "AUX", "X", "Y", "Z")):
            remapping_dict[ch_name] = "misc"
        else:
            remapping_dict[ch_name] = "misc"
    vhdr_raw.set_channel_types(remapping_dict, verbose=False)
    return vhdr_raw


def write_bids_example():
    # define run file to read and write from
    PATH_RUN = r"C:\code\py_neuromodulation\examples\data\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-3_ieeg.vhdr"
    PATH_BIDS = r"C:\code\py_neuromodulation\examples\data"
    PATH_OUT = r"C:\Users\ICN_admin\Downloads\BIDS_EXAMPLE"
    PATH_OUT_TEMP = r"C:\Users\ICN_admin\Downloads\BIDS_EXAMPLE_TEMP"

    entities = mne_bids.get_entities_from_fname(PATH_RUN)

    bids_path = mne_bids.BIDSPath(
        subject=entities["subject"],
        session=entities["session"],
        task=entities["task"],
        run=entities["run"],
        acquisition=entities["acquisition"],
        datatype="ieeg",
        root=PATH_BIDS,
    )

    raw_arr = mne_bids.read_raw_bids(bids_path)

    # crop data
    raw_arr.crop(0, 19)  # pick three movements

    data = raw_arr.get_data()[:10, :]
    ch_names = raw_arr.ch_names[:10]

    pybv.write_brainvision(
        data=data,
        sfreq=raw_arr.info["sfreq"],
        ch_names=ch_names,
        fname_base="example",
        folder_out=PATH_OUT_TEMP,
    )

    data_to_write = mne.io.read_raw_brainvision(PurePath(PATH_OUT_TEMP, "example.vhdr"))

    # example.eeg / .vhdr need to be deleted afterwards

    data_to_write = set_chtypes(data_to_write)
    data_to_write.info["line_freq"] = 60

    mne_bids.write_raw_bids(
        data_to_write,
        mne_bids.BIDSPath(
            subject="testsub",
            session="EphysMedOff",
            task="gripforce",
            datatype="ieeg",
            run="0",
            root=PATH_OUT,
        ),
        overwrite=True,
    )
