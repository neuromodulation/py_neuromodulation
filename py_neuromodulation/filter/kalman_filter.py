import numpy as np
from typing import TYPE_CHECKING


from py_neuromodulation.utils.types import NMBaseModel
from py_neuromodulation.utils.pydantic_extensions import NMErrorList


if TYPE_CHECKING:
    from py_neuromodulation.stream.settings import NMSettings


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

    def validate_fbands(self, settings: "NMSettings") -> NMErrorList:
        errors: NMErrorList = NMErrorList()

        if not all(
            (item in settings.frequency_ranges_hz for item in self.frequency_bands)
        ):
            errors.add_error(
                "Frequency bands for Kalman filter must also be specified in "
                "frequency_ranges_hz.",
                location=[
                    "kalman_filter_settings",
                    "frequency_bands",
                ],
            )

        return errors


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
    from .kalman_filter_external import KalmanFilter

    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([0, 1])  # x here sensor signal and it's first derivative
    f.F = np.array([[1, Tp], [0, 1]])
    f.H = np.array([[1, 0]])
    f.R = sigma_v
    f.Q = np.array(
        [
            [(sigma_w**2) * (Tp**3) / 3, (sigma_w**2) * (Tp**2) / 2],
            [(sigma_w**2) * (Tp**2) / 2, (sigma_w**2) * Tp],
        ]
    )
    f.P = np.cov([[1, 0], [0, 1]])
    return f
