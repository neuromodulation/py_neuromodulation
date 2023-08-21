import math
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from py_neuromodulation.nm_rereference import ReReferencer
from py_neuromodulation import (
    nm_generator,
    nm_settings,
    nm_IO,
    nm_define_nmchannels,
)

@pytest.fixture
def setup():
    """This test function sets a data batch and automatic initialized M1 datafram

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing
        settings_wrapper (settings.py): settings.json
        fs (float): example sampling frequency
    """

    RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT, datatype = nm_IO.get_paths_example_data()

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

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(
        settings
    )

    generator = nm_generator.raw_data_generator(
        data, settings, math.floor(sfreq)
    )
    data_batch = next(generator, None)
    
    return [
        raw.ch_names,
        raw.get_channel_types(),
        raw.info["bads"],
        data_batch
    ]

def test_rereference_not_used_channels_no_reref(setup):

    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )
    
    re_referencer = ReReferencer(1, nm_channels)
    ref_dat = re_referencer.process(data_batch)

    for no_ref_idx in np.where(
        (nm_channels.rereference == "None") & nm_channels.used
        == 1
    )[0]:
        assert_allclose(
            ref_dat[no_ref_idx, :], data_batch[no_ref_idx, :]
        )

def test_rereference_car(setup):

    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )
    
    re_referencer = ReReferencer(1, nm_channels)
    ref_dat = re_referencer.process(data_batch)

    for ecog_ch_idx in np.where(
        (nm_channels["type"] == "ecog")
        & (nm_channels.rereference == "average")
    )[0]:
        assert_allclose(
            ref_dat[ecog_ch_idx, :],
            data_batch[ecog_ch_idx, :]
            - data_batch[
                (nm_channels["type"] == "ecog")
                & (nm_channels.index != ecog_ch_idx)
            ].mean(axis=0),
        )

def test_rereference_bp(setup):

    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )
    
    re_referencer = ReReferencer(1, nm_channels)
    ref_dat = re_referencer.process(data_batch)

    for bp_reref_idx in [
        ch_idx
        for ch_idx, ch in enumerate(nm_channels.rereference)
        if ch in list(nm_channels.name)
    ]:
        # bp_reref_idx is the channel index of the rereference anode
        # referenced_bp_channel is the channel index which is the rereference cathode
        referenced_bp_channel = np.where(
            nm_channels.iloc[bp_reref_idx]["rereference"]
            == nm_channels.name
        )[0][0]
        assert_allclose(
            ref_dat[bp_reref_idx, :],
            data_batch[bp_reref_idx, :]
            - data_batch[referenced_bp_channel, :],
        )

def test_rereference_wrong_rererference_column_name(setup):
    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("SQUARED_ROTATION",),
    )
    
    nm_channels.loc[0, "rereference"] = "hallo"
    with pytest.raises(Exception) as e_info:
        re_referencer = ReReferencer(1, nm_channels)

def test_rereference_muliple_channels(setup):

    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )
    
    nm_channels.loc[0, "rereference"] = "LFP_RIGHT_1&LFP_RIGHT_2"

    re_referencer = ReReferencer(1, nm_channels)
    ref_dat = re_referencer.process(data_batch)

    assert_allclose(
        ref_dat[0, :], 
        data_batch[0, :] - (data_batch[1, :] + data_batch[2, :])/2
    )

def test_rereference_same_channel(setup):

    ch_names, ch_types, bads, data_batch = setup

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    nm_channels.loc[0, "rereference"] = nm_channels.loc[0, "name"]
    
    with pytest.raises(Exception):
        re_referencer = ReReferencer(1, nm_channels)
    