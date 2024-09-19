import numpy as np
from numpy.testing import assert_allclose
import pytest

import py_neuromodulation as nm
from py_neuromodulation.processing import ReReferencer


def test_rereference_not_used_channels_no_reref(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    re_referencer = ReReferencer(1, channels)

    # select here data that will is selected, this operation takes place in the nm_run_analysis
    data_used = data_batch[channels["used"] == 1]

    ref_dat = re_referencer.process(data_used)

    for no_ref_idx in np.where((channels.rereference == "None") & channels.used == 1)[
        0
    ]:
        assert_allclose(ref_dat[no_ref_idx, :], data_batch[no_ref_idx, :])


def test_rereference_car(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    re_referencer = ReReferencer(1, channels)

    data_used = data_batch[channels["used"] == 1]

    ref_dat = re_referencer.process(data_used)

    for ecog_ch_idx in np.where(
        (channels["type"] == "ecog") & (channels.rereference == "average")
    )[0]:
        assert_allclose(
            ref_dat[ecog_ch_idx, :],
            data_batch[ecog_ch_idx, :]
            - data_batch[
                (channels["type"] == "ecog") & (channels.index != ecog_ch_idx)
            ].mean(axis=0),
        )


def test_rereference_bp(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    re_referencer = ReReferencer(1, channels)

    data_used = data_batch[channels["used"] == 1]

    ref_dat = re_referencer.process(data_used)

    for bp_reref_idx in [
        ch_idx
        for ch_idx, ch in enumerate(channels.rereference)
        if ch in list(channels.name)
    ]:
        # bp_reref_idx is the channel index of the rereference anode
        # referenced_bp_channel is the channel index which is the rereference cathode
        referenced_bp_channel = np.where(
            channels.iloc[bp_reref_idx]["rereference"] == channels.name
        )[0][0]
        assert_allclose(
            ref_dat[bp_reref_idx, :],
            data_batch[bp_reref_idx, :] - data_batch[referenced_bp_channel, :],
        )


def test_rereference_wrong_rererference_column_name(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("SQUARED_ROTATION",),
    )

    channels.loc[0, "rereference"] = "hallo"
    with pytest.raises(Exception):
        ReReferencer(1, channels)


def test_rereference_muliple_channels(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    channels.loc[0, "rereference"] = "LFP_RIGHT_1&LFP_RIGHT_2"

    re_referencer = ReReferencer(1, channels)

    data_used = data_batch[channels["used"] == 1]

    ref_dat = re_referencer.process(data_used)

    assert_allclose(
        ref_dat[0, :],
        data_batch[0, :] - (data_batch[1, :] + data_batch[2, :]) / 2,
    )


def test_rereference_same_channel(setup_databatch):
    ch_names, ch_types, bads, data_batch = setup_databatch

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=bads,
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT",),
    )

    channels.loc[0, "rereference"] = channels.loc[0, "name"]

    with pytest.raises(Exception):
        ReReferencer(1, channels)
