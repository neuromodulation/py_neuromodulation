import numpy as np
from mne_connectivity import make_signals_in_freq_bands

import py_neuromodulation as nm


def test_coherence():
    """Check that coherence features compute properly and match expected values."""
    # Simulate connectivity data (interaction at specified frequency band)
    sfreq = 500  # Hz
    n_epochs = 1
    n_times = sfreq * 10  # samples
    fband = (15, 20)  # frequency band of interaction, Hz
    trans = 2  # transition bandwidth of signal, Hz
    delay = 50  # samples
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
        snr=0.9,  # change here requires change in `signal/noise_con` vars below
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
    settings.features.coherence = True

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
    settings.coherence_settings.frequency_bands = ["signal", "noise_low", "noise_high"]
    settings.coherence_settings.nperseg = sfreq
    # only average within each band required
    settings.coherence_settings.features = {
        "mean_fband": True, "max_fband": False, "max_allfbands": False
    }

    # unique all-to-all connectivity indices, i.e.: ([0], [1])
    # XXX: avoids pydantic ValidationError that lists are too short (length == 1)
    settings.coherence_settings.channels = [ch_names]

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
        experiment_name="test_coherence",
    )

    # Aggregate results over windows
    results = {key: None for key in features.keys()}
    results.pop("time")
    for key in results.keys():
        # average over windows; take absolute before averaging icoh values
        results[key] = np.abs(features[key].values).mean()

    node_name = "seed_to_target"
    #for con_method in ["coh", "icoh"]:
    con_method = "icoh"
    # Assert that frequencies of simulated interaction have strong connectivity
    assert (
        results[f"{con_method}_{node_name}_mean_fband_signal"] >
        results[f"{con_method}_{node_name}_mean_fband_noise_low"]
    )

    assert (
        results[f"{con_method}_{node_name}_mean_fband_signal"] >
        results[f"{con_method}_{node_name}_mean_fband_noise_high"]
    )
