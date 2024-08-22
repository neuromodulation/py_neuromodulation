import numpy as np
from typing import cast

from py_neuromodulation.utils.types import NMPreprocessor
from py_neuromodulation import logger


class NotchFilter(NMPreprocessor):
    def __init__(
        self,
        sfreq: float,
        line_noise: float | None = None,
        freqs: np.ndarray | None = None,
        notch_widths: int | np.ndarray | None = 3,
        trans_bandwidth: float = 6.8,
    ) -> None:
        from mne.filter import create_filter

        if line_noise is None and freqs is None:
            raise ValueError(
                "Either line_noise or freqs must be defined if notch_filter is"
                "activated."
            )

        if freqs is None:
            freqs = np.arange(line_noise, sfreq / 2, line_noise, dtype=int)

        if freqs.size > 0 and freqs[-1] >= sfreq / 2:
            freqs = freqs[:-1]

        # Code is copied from filter.py notch_filter
        if freqs.size == 0:
            self.filter_bank = None
            logger.warning(
                "WARNING: notch_filter is activated but data is not being"
                " filtered. This may be due to a low sampling frequency or"
                " incorrect specifications. Make sure your settings are"
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
                    "notch_widths must be None, scalar, or the " "same length as freqs"
                )
        notch_widths = cast(np.ndarray, notch_widths)  # For MyPy only, no runtime cost

        # Speed this up by computing the fourier coefficients once
        tb_half = trans_bandwidth / 2.0
        lows = [freq - nw / 2.0 - tb_half for freq, nw in zip(freqs, notch_widths)]
        highs = [freq + nw / 2.0 + tb_half for freq, nw in zip(freqs, notch_widths)]

        self.filter_bank = create_filter(
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
            verbose=False,
        )

    def process(self, data: np.ndarray) -> np.ndarray:
        if self.filter_bank is None:
            return data

        from mne.filter import _overlap_add_filter

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
