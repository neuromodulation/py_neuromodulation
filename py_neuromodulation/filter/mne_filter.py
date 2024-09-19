import numpy as np
from collections.abc import Sequence


class MNEFilter:
    """mne.filter wrapper

    This class stores for given frequency band ranges the filter
    coefficients with length "filter_len".
    The filters can then be used sequentially for band power estimation with
    apply_filter().
    Note that this filter can be a bandpass, bandstop, lowpass, or highpass filter
    depending on the frequency ranges given (see further details in mne.filter.create_filter).

    Parameters
    ----------
    f_ranges : list[tuple[float | None, float | None]]
    sfreq : float
        Sampling frequency.
    filter_length : str, optional
        Filter length. Human readable (e.g. "1000ms", "1s"), by default "999ms"
    l_trans_bandwidth : float | str, optional
        Length of the lower transition band or "auto", by default 4
    h_trans_bandwidth : float | str, optional
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
        f_ranges: Sequence[tuple[float | None, float | None]],
        sfreq: float,
        filter_length: str | float = "999ms",
        l_trans_bandwidth: float | str = 4,
        h_trans_bandwidth: float | str = 4,
        verbose: bool | int | str | None = None,
    ) -> None:
        from mne.filter import create_filter

        filter_bank = []
        # mne create_filter function only accepts str and int for filter_length
        if isinstance(filter_length, float):
            filter_length = int(filter_length)

        for f_range in f_ranges:
            try:
                filt = create_filter(
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
            except ValueError:
                filt = create_filter(
                    None,
                    sfreq,
                    l_freq=f_range[0],
                    h_freq=f_range[1],
                    fir_design="firwin",
                    verbose=verbose,
                    # filter_length=filter_length,
                )
            filter_bank.append(filt)

        self.num_filters = len(filter_bank)
        self.filter_bank = np.vstack(filter_bank)

        self.filters: np.ndarray
        self.num_channels = -1

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

        """
        from scipy.signal import fftconvolve

        if data.ndim > 2:
            raise ValueError(
                f"Data must have one or two dimensions. Got:"
                f" {data.ndim} dimensions."
            )
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if self.num_channels == -1:
            self.num_channels = data.shape[0]
            self.filters = np.tile(
                self.filter_bank[None, :, :], (self.num_channels, 1, 1)
            )

        data_tiled = np.tile(data[:, None, :], (1, self.num_filters, 1))

        filtered = fftconvolve(data_tiled, self.filters, axes=2, mode="same")

        # ensure here that the output dimension matches the input dimension
        if data.shape[1] != filtered.shape[-1]:
            # select the middle part of the filtered data
            middle_index = filtered.shape[-1] // 2
            filtered = filtered[
                :,
                :,
                middle_index - data.shape[1] // 2 : middle_index + data.shape[1] // 2,
            ]

        return filtered
