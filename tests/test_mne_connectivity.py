from typing import Literal

import numpy as np
import pytest
from mne_connectivity import make_signals_in_freq_bands

import py_neuromodulation as nm


@pytest.mark.parametrize("method", ["coh", "dpli"])
def test_mne_connectivity(method: Literal["coh", "dpli"]):
    """Check that mne_connectivity features compute properly and match expected values."""
    # Simulate connectivity data (interaction at specified frequency band)
    sfreq = 500  # Hz
    n_epochs = 1
    n_times = sfreq * 2  # samples
    fband = (15, 20)  # frequency band of interaction, Hz
    trans = 2  # transition bandwidth of signal, Hz
    delay = 5  # samples; seed leads target to test directed connectivity methods
    epochs = make_signals_in_freq_bands(
        n_seeds=1,
        n_targets=1,
        freq_band=fband,
        n_epochs=n_epochs,
        n_times=n_times,
        sfreq=sfreq,
        trans_bandwidth=trans,
        connection_delay=delay,
        ch_names=["seed", "target"],
        snr=0.7,  # change here requires change in `signal/noise_con` vars below
        rng_seed=44
    )

    # Set up py_nm channels info
    ch_names = epochs.ch_names
    ch_types = epochs.get_channel_types()
    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=None,
        new_names="default",
        used_types=tuple(np.unique(ch_types)),
        target_keywords=None,
    )

    # Set up pn_nm processing settings
    settings = nm.NMSettings.get_default()
    settings.reset()
    settings.features.mne_connectivity = True

    # redefine freq. bands of interest
    # (accounts for signal-noise transition bandwdith when defining frequencies)
    settings.frequency_ranges_hz = {
        "signal": {  # strong connectivity expected
            "frequency_low_hz": fband[0],
            "frequency_high_hz": fband[1],
        },
        "noise_low": {  # weak connectivity expected from 1 Hz to start of interaction
            "frequency_low_hz": 1,
            "frequency_high_hz": fband[0] - trans * 2,
        },
        "noise_high": {  # weak connectivity expected from end of interaction to Nyquist
            "frequency_low_hz": fband[1] + trans * 2,
            "frequency_high_hz": sfreq // 2 - 1,
        },
    }

    settings.mne_connectivity_settings.method = method

    # unique all-to-all connectivity indices
    if method == "coh":  # undirected, so unique combination is [[0, 1]]
        settings.mne_connectivity_settings.channels = [ch_names]
    else:  # directed, so unique combinations are [[0, 1], [1, 0]]
        settings.mne_connectivity_settings.channels = [ch_names, ch_names[::-1]]

    # do not normalise features for this test!
    # (normalisation changes interpretability of connectivity values, making it harder to
    # define 'expected' connectivity values)
    settings.postprocessing.feature_normalization = False

    # Set up py_nm stream
    stream = nm.Stream(
        settings=settings,
        channels=channels,
        path_grids=None,
        verbose=True,
        sfreq=epochs.info["sfreq"],
    )

    # Compute connectivity
    features = stream.run(
        epochs.get_data(copy=False)[0],  # extract first (and only) epoch from obj
        out_dir="./test_data",
        experiment_name="test_mne_connectivity",
    )

    # Aggregate results over windows
    results = {key: None for key in features.keys()}
    results.pop("time")
    for key in results.keys():
        # average over windows; take absolute before averaging icoh values
        results[key] = np.abs(features[key].values).mean()

    if method == "coh":
        # Define expected connectivity values for signal and noise frequencies
        noise_con = 0.4
        signal_con = 0.8

        # Assert that frequencies of simulated interaction have strong connectivity
        np.testing.assert_array_less(
            signal_con, results["coh_seed_to_target_mean_fband_signal"]
        )
        # Assert that frequencies of noise have weak connectivity
        np.testing.assert_array_less(
            results["coh_seed_to_target_mean_fband_noise_low"], noise_con
        )
        np.testing.assert_array_less(
            results["coh_seed_to_target_mean_fband_noise_high"], noise_con
        )
    else:
        # Define expected connectivity values for signal and noise frequencies
        seed_leading = 0.65
        target_leading = 0.35

        # Assert that frequencies of simulated interaction have strong connectivity, i.e.:
        # >> 0.5 for seed-to-target; << 0.5 for target-to-seed
        np.testing.assert_array_less(
            seed_leading, results["dpli_seed_to_target_mean_fband_signal"]
        )
        np.testing.assert_array_less(
            results["dpli_target_to_seed_mean_fband_signal"], target_leading
        )
        # Assert that frequencies of noise have weak connectivity (i.e., ~0.5)
        for con_name in ["seed_to_target", "target_to_seed"]:
            np.testing.assert_array_less(
                results[f"dpli_{con_name}_mean_fband_noise_low"], seed_leading
            )
            np.testing.assert_array_less(
                target_leading, results[f"dpli_{con_name}_mean_fband_noise_low"]
            )
