import numpy as np

from py_neuromodulation import (
    nm_define_nmchannels,
    nm_IO,
    nm_settings,
    Stream,
)


def test_bispectrum():
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
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype)

    ch_names = raw.ch_names[4]
    ch_types = raw.get_channel_types()[4]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=[ch_names],
        ch_types=[ch_types],
        reference="default",
        bads=None,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT_CLEAN",),
    )

    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)

    settings["features"]["bispectrum"] = True

    stream = Stream(
        settings=settings,
        nm_channels=nm_channels,
        path_grids=None,
        verbose=True,
        sfreq=sfreq,
        line_noise=line_noise,
        coord_list=coord_list,
        coord_names=coord_names,
    )

    features = stream.run(np.expand_dims(data[3, :], axis=0))

    assert features["ECOG_RIGHT_1_Bispectrum_phase_mean_whole_fband_range"].sum() != 0
