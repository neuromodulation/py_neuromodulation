"""Module for resampling."""
### Importing the function directly instead of the entire package saves less
### time than expected, and is usually not recommended. And this way the code
### becomes more readable.
import mne
import numpy as np


class Resample:
    ### According to numpy documentation, the Attributes as well as Parameters
    ### for class construction (__init__) should be documented here.
    """Resample data.

    Parameters
    ----------
    sfreq_old : int | float
        Original sampling frequency.
    sfreq_new : int | float
        New sampling frequency.

    Attributes
    ----------
    up: float
        Factor to upsample by.
    """

    ### I renamed fs to sfreq because this is the MNE nomenclature and we are
    ### heavily using MNE, but of course if you don't agree we don't have to
    def __init__(self, sfreq_old: int | float, sfreq_new: int | float) -> None:
        ### Settings is only used once, so we can just pass sampling frequency
        ### to the class directly
        ratio = sfreq_new / sfreq_old
        if ratio == 1.0:
            self.up = 0.0
        else:
            self.up = ratio

    def resample_raw(self, data: np.ndarray) -> np.ndarray:
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
        return mne.filter.resample(data, up=self.up, down=1.0)
