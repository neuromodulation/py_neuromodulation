import pytest
import numpy as np

import py_neuromodulation as nm


@pytest.fixture
def setup_default_data():
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

    return raw, data, sfreq


@pytest.fixture
def setup_default_stream_fast_compute() -> tuple[np.ndarray, nm.Stream]:
    """This test function sets a data batch and automatic initialized M1 dataframe.
    Only fft features is enabled.

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing
        settings_wrapper (settings.py): settings.json
        fs (float): example sampling frequency
    """

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
        target_keywords=("MOV_RIGHT_CLEAN",),
    )

    settings = nm.NMSettings.get_default()
    settings.reset()
    settings.features.fft = True

    stream = nm.Stream(
        settings=settings,
        channels=channels,
        path_grids=None,
        verbose=True,
        sfreq=sfreq,
        line_noise=line_noise,
        coord_list=coord_list,
        coord_names=coord_names,
    )

    return data, stream

@pytest.fixture
def setup_default_stream_fooof() -> tuple[np.ndarray, nm.Stream]:
    """This test function sets a data batch and automatic initialized M1 dataframe.
    Only fooof feature is enabled.

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing
        settings_wrapper (settings.py): settings.json
        fs (float): example sampling frequency
    """

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
        target_keywords=("MOV_RIGHT_CLEAN",),
    )

    settings = nm.NMSettings.get_default()
    settings.reset()
    settings.fooof_settings.aperiodic.exponent = True
    settings.fooof_settings.aperiodic.offset = True
    settings.features.fooof = True

    stream = nm.Stream(
        settings=settings,
        channels=channels,
        path_grids=None,
        verbose=True,
        sfreq=sfreq,
        line_noise=line_noise,
        coord_list=coord_list,
        coord_names=coord_names,
    )

    return data, stream


@pytest.fixture
def setup_lsl_player(request):
    """This test function sets a data batch and automatic initialized dataframe

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        player (mne_lsl.player.PlayerLSL): LSL player object
    """

    name = request.param
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

    player = nm.stream.LSLOfflinePlayer(
        raw=raw,
        stream_name=name,
    )
    return player


@pytest.fixture
def setup_databatch():
    """This test function sets a data batch and automatic initialized M1 dataframe

    Args:
        PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

    Returns:
        ieeg_batch (np.ndarray): (channels, samples)
        df_M1 (pd Dataframe): auto intialized table for rereferencing
        settings_wrapper (settings.py): settings.json
        fs (float): example sampling frequency
    """

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

    settings = nm.NMSettings.get_fast_compute()

    generator = nm.stream.RawDataGenerator(
        data,
        int(np.floor(sfreq)),
        settings.sampling_rate_features_hz,
        settings.segment_length_features_ms,
    )
    data_batch = next(generator, None)

    return [raw.ch_names, raw.get_channel_types(), raw.info["bads"], data_batch[1]]
