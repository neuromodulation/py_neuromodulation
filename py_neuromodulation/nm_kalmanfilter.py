from numpy import array, cov
from py_neuromodulation.nm_types import NMBaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


class KalmanSettings(NMBaseModel):
    Tp: float = 0.1
    sigma_w: float = 0.7
    sigma_v: float = 1.0
    frequency_bands: list[str] = [
        "theta",
        "alpha",
        "low_beta",
        "high_beta",
        "low_gamma",
        "high_gamma",
        "HFA",
    ]

    def validate_fbands(self, settings: "NMSettings") -> None:
        assert all(
            (item in settings.frequency_ranges_hz for item in self.frequency_bands)
        ), (
            "Frequency bands for Kalman filter must also be specified in "
            "bandpass_filter_settings."
        )


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
    from filterpy.kalman import KalmanFilter

    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = array([0, 1])  # x here sensor signal and it's first derivative
    f.F = array([[1, Tp], [0, 1]])
    f.H = array([[1, 0]])
    f.R = sigma_v
    f.Q = array(
        [
            [(sigma_w**2) * (Tp**3) / 3, (sigma_w**2) * (Tp**2) / 2],
            [(sigma_w**2) * (Tp**2) / 2, (sigma_w**2) * Tp],
        ]
    )
    f.P = cov([[1, 0], [0, 1]])
    return f
