"""Module for filter functionality."""
import mne
from mne.filter import _overlap_add_filter
import numpy as np


class BandPassFilter:
    """Bandpass filters data in given frequency ranges.

    This class stores for given frequency band ranges the filter
    coefficients with length "filter_len".
    The filters can then be used sequentially for band power estimation with
    apply_filter().

    Parameters
    ----------
    f_ranges : list of lists
        Frequency ranges. Inner lists must be of length 2.
    sfreq : int | float
        Sampling frequency.
    filter_length : str, optional
        Filter length. Human readable (e.g. "1000ms", "1s"), by default "999ms"
    l_trans_bandwidth : int | float | str, optional
        Length of the lower transition band or "auto", by default 4
    h_trans_bandwidth : int | float | str, optional
        Length of the higher transition band or "auto", by default 4
    verbose : bool | None, optional
        Verbosity level, by default None

    Attributes
    ----------
    filter_bank: np.ndarray shape (n,)
        Factor to upsample by.
    """

    def __init__(
        self,
        f_ranges: list[list[int | float | None]],
        sfreq: int | float,
        filter_length: str | float = "999ms",
        l_trans_bandwidth: int | float | str = 4,
        h_trans_bandwidth: int | float | str = 4,
        verbose: bool | int | str | None = None,
    ) -> None:
        filter_bank = []
        # mne create_filter function only accepts str and int
        if isinstance(filter_length, float):
            filter_length = int(filter_length)

        for f_range in f_ranges:
            filt = mne.filter.create_filter(
                None,
                sfreq,
                l_freq=f_range[0],
                h_freq=f_range[1],
                fir_design="firwin",
                l_trans_bandwidth=l_trans_bandwidth,  # type: ignore
                h_trans_bandwidth=h_trans_bandwidth,  # type: ignore
                filter_length=filter_length,  # type: ignore
                verbose=verbose,
            )
            filter_bank.append(filt)
        self.filter_bank = np.vstack(filter_bank)

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """Apply previously calculated (bandpass) filters to data.

        Parameters
        ----------
        data : np.ndarray (n_samples, ) or (n_channels, n_samples)
            Data to be filtered
        filter_bank : np.ndarray, shape (n_fbands, filter_len)
            Output of calc_bandpass_filters.

        Returns
        -------
        np.ndarray, shape (n_channels, n_fbands, n_samples)
            Filtered data.

        Raises
        ------
        ValueError
            If data.ndim > 2
        """
        if data.ndim > 2:
            raise ValueError(
                f"Data must have one or two dimensions. Got:"
                f" {data.ndim} dimensions."
            )
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        filtered = np.array(
            [
                [
                    np.convolve(flt, chan, mode="same")
                    for flt in self.filter_bank
                ]
                for chan in data
            ]
        )
        return filtered


class NotchFilter:
    def __init__(
        self,
        sfreq: int | float,
        line_noise: int | float | None = None,
        freqs: np.ndarray | None = None,
        notch_widths: int | np.ndarray | None = 3,
        trans_bandwidth: int = 6.8,
    ) -> None:
        if line_noise is None and freqs is None:
            raise ValueError(
                "Either line_noise or freqs must be defined if notch_filter is"
                "activated."
            )
        if freqs is None:
            freqs = np.arange(line_noise, sfreq / 2, line_noise, dtype=int)

        if freqs.size > 0:
            if freqs[-1] >= sfreq / 2:
                freqs = freqs[:-1]

        # Code is copied from filter.py notch_filter
        if freqs.size == 0:
            self.filter_bank = None
            print(
                "WARNING: notch_filter is activated but data is not being"
                f" filtered. This may be due to a low sampling frequency or"
                f" incorrect specifications. Make sure your settings are"
                f" correct. Got: {sfreq = }, {line_noise = }, {freqs = }."
            )
            return

        filter_length = int(sfreq - 1)
        if notch_widths is None:
            notch_widths = freqs / 200.0
        elif np.any(notch_widths < 0):
            raise ValueError("notch_widths must be >= 0")
        else:
            notch_widths = np.atleast_1d(notch_widths)
            if len(notch_widths) == 1:
                notch_widths = notch_widths[0] * np.ones_like(freqs)
            elif len(notch_widths) != len(freqs):
                raise ValueError(
                    "notch_widths must be None, scalar, or the "
                    "same length as freqs"
                )

        # Speed this up by computing the fourier coefficients once
        tb_half = trans_bandwidth / 2.0
        lows = [
            freq - nw / 2.0 - tb_half for freq, nw in zip(freqs, notch_widths)
        ]
        highs = [
            freq + nw / 2.0 + tb_half for freq, nw in zip(freqs, notch_widths)
        ]

        self.filter_bank = mne.filter.create_filter(
            data=None,
            sfreq=sfreq,
            l_freq=highs,
            h_freq=lows,
            filter_length=filter_length,  # type: ignore
            l_trans_bandwidth=tb_half,  # type: ignore
            h_trans_bandwidth=tb_half,  # type: ignore
            method="fir",
            iir_params=None,
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            verbose=False
        )

    def process(self, data: np.ndarray) -> np.ndarray:
        if self.filter_bank is None:
            return data
        return _overlap_add_filter(
                x=data,
                h=self.filter_bank,
                n_fft=None,
                phase="zero",
                picks=None,
                n_jobs=1,
                copy=True,
                pad="reflect_limited",
            )
