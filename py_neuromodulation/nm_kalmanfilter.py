from numpy import array, cov
from pydantic.dataclasses import dataclass
from pydantic import Field

from filterpy.kalman import KalmanFilter


@dataclass
class KalmanSettings:
    @staticmethod
    def default_bands() -> list[str]:
        return [
            "theta",
            "alpha",
            "low_beta",
            "high_beta",
            "low_gamma",
            "high_gamma",
            "HFA",
        ]

    Tp: float = 0.1
    sigma_w: float = 0.7
    sigma_v: float = 1.0
    frequency_bands: list[str] = Field(default_factory=default_bands, min_length=1)


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
    f.Q = array(
        [
            [(sigma_w**2) * (Tp**3) / 3, (sigma_w**2) * (Tp**2) / 2],
            [(sigma_w**2) * (Tp**2) / 2, (sigma_w**2) * Tp],
        ]
    )
    f.P = cov([[1, 0], [0, 1]])
    return f
