"""Module for handling settings."""

from py_neuromodulation import PYNM_DIR
from py_neuromodulation.nm_IO import read_settings


def get_default_settings() -> dict:
    """Read default settings from nm_settings.json"""
    settings_path = PYNM_DIR / "nm_settings.json"
    return read_settings(settings_path)


def reset_settings(
    settings: dict,
) -> dict:
    for f in settings["features"]:
        settings["features"][f] = False
    settings["preprocessing"] = []
    for f in settings["postprocessing"]:
        settings["postprocessing"][f] = False
    return settings


def set_settings_fast_compute(
    settings: dict,
) -> dict:
    settings = reset_settings(settings)
    settings["features"]["fft"] = True
    settings["preprocessing"] = [
        "raw_resampling",
        "notch_filter",
        "re_referencing",
    ]
    settings["postprocessing"]["feature_normalization"] = True
    settings["postprocessing"]["project_cortex"] = False
    settings["postprocessing"]["project_subcortex"] = False
    return settings


def test_settings(
    settings: dict,
) -> None:
    """Test if settings are specified correctly in nm_settings.json
    Parameters
    ----------
    settings: dict
        settings to tests
    verbose: boolean
        set to True if feedback is desired.
    Returns
    -------
    None
    """
    s = settings

    assert isinstance(s["sampling_rate_features_hz"], (float, int))

    assert (
        isinstance(value, bool) for value in s["features"].values()
    ), "features must be a boolean value."

    assert any(
        value is True for value in s["features"].values()
    ), "Set at least one feature to True."
