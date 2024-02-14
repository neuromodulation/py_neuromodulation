import pytest
import numpy as np

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

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    generator = nm_generator.raw_data_generator(
        data, settings, int(np.floor(sfreq))
    )
    data_batch = next(generator, None)

    return [raw.ch_names, raw.get_channel_types(), raw.info["bads"], data_batch]
