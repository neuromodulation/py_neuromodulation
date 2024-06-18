import pytest
import numpy as np

from py_neuromodulation import (
    nm_generator,
    nm_stream,
    nm_settings,
    nm_mnelsl_generator,
    nm_IO,
    nm_define_nmchannels,
    NMSettings
)

@pytest.fixture
def setup_default_data():
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
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN)
    
    return raw, data, sfreq

@pytest.fixture
def setup_default_stream_fast_compute() -> tuple[np.ndarray, nm_stream.Stream]:
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
    ) = nm_IO.get_paths_example_data()

    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN)

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("MOV_RIGHT_CLEAN",),
    )

    settings = NMSettings.get_default()
    settings.reset()
    settings.fooof.aperiodic.exponent = True
    settings.fooof.aperiodic.offset = True
    settings.features.fooof = True

    stream = nm_stream.Stream(
        settings=settings,
        nm_channels=nm_channels,
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
    """ This test function sets a data batch and automatic initialized dataframe 
    
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
    ) = nm_IO.get_paths_example_data()
    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN)
    player = nm_mnelsl_generator.LSLOfflinePlayer(raw = raw, stream_name=name)
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
    ) = nm_IO.get_paths_example_data()

    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN)

    settings = NMSettings.get_fast_compute()

    generator = nm_generator.raw_data_generator(data, settings, int(np.floor(sfreq)))
    data_batch = next(generator, None)

    return [raw.ch_names, raw.get_channel_types(), raw.info["bads"], data_batch[1]]
