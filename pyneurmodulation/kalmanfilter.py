from filterpy.kalman import KalmanFilter
import numpy as np

def define_KF(Tp, sigma_w, sigma_v):
    """
    define Kalman Filter according to White Noise Acceleratin model
    """

    f = KalmanFilter (dim_x=2, dim_z=1)
    f.x = np.array([0, 1])# x here sensor signal and it's first derivative
    f.F = np.array([[1, Tp], [0, 1]])
    f.H = np.array([[1, 0]])
    f.R = sigma_v
    f.Q = np.array([[(sigma_w**2)*(Tp**3)/3, (sigma_w**2)*(Tp**2)/2],\
                    [(sigma_w**2)*(Tp**2)/2, (sigma_w**2)*Tp]])
    f.P = np.cov([[1, 0], [0, 1]]) 
    return f