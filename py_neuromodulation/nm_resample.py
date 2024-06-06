"""Module for resampling."""

import numpy as np
from py_neuromodulation.nm_types import NMBaseModel, Field

from mne.filter import resample as mne_resample

from py_neuromodulation.nm_preprocessing import NMPreprocessor


class ResamplerSettings(NMBaseModel):
    resample_freq_hz: float = Field(default=1000, gt=0)


class Resampler(NMPreprocessor):
    """Resample data.

    Parameters
    ----------
    sfreq : float
        Original sampling frequency.

    Attributes
    ----------
    up: float
        Factor to upsample by.
    """

    def __init__(
        self,
        sfreq: float,
        resample_freq_hz: float,
    ) -> None:
        self.settings = ResamplerSettings(resample_freq_hz=resample_freq_hz)

        ratio = float(resample_freq_hz / sfreq)
        if ratio == 1.0:
            self.up = 0.0
        else:
            self.up = ratio

    def process(self, data: np.ndarray) -> np.ndarray:
        """Resample raw data using mne.filter.resample.

        Parameters
        ----------
        data : np.ndarray
            Data to resample

        Returns
        -------
        np.ndarray
            Resampled data
        """
        if not self.up:
            return data
        return mne_resample(data.astype(np.float64), up=self.up, down=1.0)
