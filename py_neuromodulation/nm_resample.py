"""Module for resampling."""
import mne
import numpy as np

from py_neuromodulation.nm_preprocessing import NMPreprocessor

class Resampler(NMPreprocessor):
    """Resample data.

    Parameters
    ----------
    sfreq : int | float
        Original sampling frequency.

    Attributes
    ----------
    up: float
        Factor to upsample by.
    """

    def __init__(
        self,
        sfreq: int | float,
        resample_freq_hz: int | float,
    ) -> None:

        assert isinstance(resample_freq_hz, (float, int))

        self.sfreq_new: int | float = resample_freq_hz
        ratio = float(self.sfreq_new / sfreq)
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
        return mne.filter.resample(data.astype(np.float64), up=self.up, down=1.0)
