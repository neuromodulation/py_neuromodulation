from numpy import array, cov
from typing import Iterable

from filterpy.kalman import KalmanFilter


def define_KF(Tp, sigma_w, sigma_v):
    """Define Kalman filter according to white noise acceleration model.
    See DOI: 10.1109/TBME.2009.2038990  for explanation
    See https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html#r64ca38088676-2 for implementation details

    Parameters
    ----------
    Tp : float
        prediction interval
    sigma_w : float
        process noise
    sigma_v : float
        measurement noise

    Returns
    -------
    filterpy.KalmanFilter
        initialized KalmanFilter object
    """

    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = array([0, 1])  # x here sensor signal and it's first derivative
    f.F = array([[1, Tp], [0, 1]])
    f.H = array([[1, 0]])
    f.R = sigma_v
    f.Q = array([[(sigma_w**2)*(Tp**3)/3, (sigma_w**2)*(Tp**2)/2],
                 [(sigma_w**2)*(Tp**2)/2, (sigma_w**2)*Tp]])
    f.P = cov([[1, 0], [0, 1]])
    return f

def test_kf_settings(
        s: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        assert isinstance(s["kalman_filter_settings"]["Tp"], (float, int))
        assert isinstance(s["kalman_filter_settings"]["sigma_w"], (float, int))
        assert isinstance(s["kalman_filter_settings"]["sigma_v"], (float, int))
        assert s["kalman_filter_settings"][
            "frequency_bands"
        ], "No frequency bands specified for Kalman filter."
        assert isinstance(
            s["kalman_filter_settings"]["frequency_bands"], list
        ), "Frequency bands for Kalman filter must be specified as a list."
        assert (
            item
            in s["frequency_ranges_hz"].values()
            for item in s["kalman_filter_settings"]["frequency_bands"]
        ), (
            "Frequency bands for Kalman filter must also be specified in "
            "bandpass_filter_settings."
        )