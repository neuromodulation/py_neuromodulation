from numpy import array, cov

from filterpy.kalman import KalmanFilter


def define_KF(Tp, sigma_w, sigma_v):
    """
    Define Kalman filter according to white noise acceleration model.
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
