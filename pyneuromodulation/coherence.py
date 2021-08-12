"""Implements Partial Directed Coherence and Direct Transfer Function
using MVAR processes.
Reference
---------
Luiz A. Baccala and Koichi Sameshima. Partial directed coherence:
a new concept in neural structure determination.
Biological Cybernetics, 84(6):463:474, 2001.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from math import ceil as math_ceil
from math import log as math_log
from math import pi as math_pi
from numpy import argmin, concatenate, diag, dot, empty, eye, nanmean, \
    nan_to_num, ones, reshape, zeros
from numpy import abs as np_abs
from numpy import arange as np_arange
from numpy import inf as np_inf
from numpy import log as np_log
from numpy import log2 as np_log2
from numpy import sqrt as np_sqrt
from scipy.linalg import det as la_det
from scipy.linalg import inv as la_inv
from scipy.linalg import solve as la_solve
from scipy.fftpack import fft, fftfreq


def cov(X, p):
    """vector autocovariance up to order p
    Parameters
    ----------
    X : ndarray, shape (N, n)
        The N time series of length n
    p : int
        order
    Returns
    -------
    R : ndarray, shape (p + 1, N, N)
        The autocovariance up to order p
    """
    N, n = X.shape
    R = zeros((p + 1, N, N))
    for k in range(p + 1):
        R[k] = (1. / float(n - k)) * dot(X[:, :n - k], X[:, k:].T)
    return R


def mvar_fit(X, p):
    """Fit MVAR model of order p using Yule Walker
    Parameters
    ----------
    X : ndarray, shape (N, n)
        The N time series of length n
    p : int
        The order of the MVAR model
    Returns
    -------
    A : ndarray, shape (p, N, N)
        The AR coefficients where N is the number of signals
        and p the order of the model.
    sigma : array, shape (N,)
        The noise for each time series
    """
    N, n = X.shape
    gamma = cov(X, p)  # gamma(r,i,j) cov between X_i(0) et X_j(r)
    G = zeros((p * N, p * N))
    gamma2 = concatenate(gamma, axis=0)
    gamma2[:N, :N] /= 2.

    for i in range(p):
        G[N * i:, N * i:N * (i + 1)] = gamma2[:N * (p - i)]

    G = G + G.T  # big block matrix

    gamma4 = concatenate(gamma[1:], axis=0)

    phi = la_solve(G, gamma4)  # solve Yule Walker

    tmp = dot(gamma4[:N * p].T, phi)
    sigma = gamma[0] - tmp - tmp.T + dot(phi.T, dot(G, phi))

    phi = reshape(phi, (p, N, N))
    for k in range(p):
        phi[k] = phi[k].T

    return phi, sigma


def compute_order(X, p_max):
    """Estimate AR order with BIC
    Parameters
    ----------
    X : ndarray, shape (N, n)
        The N time series of length n
    p_max : int
        The maximum model order to test
    Returns
    -------
    p : int
        Estimated order
    bic : ndarray, shape (p_max + 1,)
        The BIC for the orders from 0 to p_max.
    """
    N, n = X.shape

    bic = empty(p_max + 1)
    bic[0] = np_inf

    Y = X.T

    for p in range(1, p_max + 1):
        A, sigma = mvar_fit(X, p)
        A_2d = concatenate(A, axis=1)

        n_samples = n - p
        bic[p] = n_samples * N * math_log(2. * math_pi)
        bic[p] += n_samples * np_log(la_det(sigma))
        bic[p] += p * (N ** 2) * math_log(n_samples)
        bic[p] += p * (N ** 2) * math_log(n_samples)

        sigma_inv = la_inv(sigma)
        S = 0.
        for i in range(p, n):
            res = Y[i] - dot(A_2d, Y[i - p:i][::-1, :].ravel())
            S += dot(res, sigma_inv.dot(res))

        bic[p] += S

    p = argmin(nan_to_num(bic, nan=np_inf, posinf=np_inf))
    # p = argmin(bic)
    return p, bic


def spectral_density(A, n_fft=None):
    """Estimate PSD from AR coefficients
    Parameters
    ----------
    A : ndarray, shape (p, N, N)
        The AR coefficients where N is the number of signals
        and p the order of the model.
    n_fft : int
        The length of the FFT
    Returns
    -------
    fA : ndarray, shape (n_fft, N, N)
        The estimated spectral density.
    """
    p, N, N = A.shape
    if n_fft is None:
        n_fft = max(int(2 ** math_ceil(np_log2(p))), 512)
    A2 = zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fft(A2, axis=0)
    freqs = fftfreq(n_fft)
    I_ = eye(N)

    for i in range(n_fft):
        fA[i] = la_inv(I_ - fA[i])

    return fA, freqs


def DTF(A, sigma=None, n_fft=None):
    """Direct Transfer Function (DTF)
    Parameters
    ----------
    A : ndarray, shape (p, N, N)
        The AR coefficients where N is the number of signals
        and p the order of the model.
    sigma : array, shape (N, )
        The noise for each time series
    n_fft : int
        The length of the FFT
    Returns
    -------
    D : ndarray, shape (n_fft, N, N)
        The estimated DTF
    """
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math_ceil(np_log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    D = zeros((n_fft, N, N))

    if sigma is None:
        sigma = ones(N)

    for i in range(n_fft):
        S = H[i]
        V = (S * sigma[None, :]).dot(S.T.conj())
        V = np_abs(diag(V))
        D[i] = np_abs(S * np_sqrt(sigma[None, :])) / np_sqrt(V)[:, None]

    return D, freqs


def PDC(A, sigma=None, n_fft=None):
    """Partial directed coherence (PDC)
    Parameters
    ----------
    A : ndarray, shape (p, N, N)
        The AR coefficients where N is the number of signals
        and p the order of the model.
    sigma : array, shape (N,)
        The noise for each time series.
    n_fft : int
        The length of the FFT.
    Returns
    -------
    P : ndarray, shape (n_fft, N, N)
        The estimated PDC.
    """
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math_ceil(np_log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    P = zeros((n_fft, N, N))

    if sigma is None:
        sigma = ones(N)

    for i in range(n_fft):
        B = H[i]
        B = la_inv(B)
        V = np_abs(dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = diag(V)  # denominator squared
        P[i] = np_abs(B * (1. / np_sqrt(sigma))[None, :]) / np_sqrt(V)[None, :]

    return P, freqs


def get_pdc(features_, s, data, filt, chs) -> dict:

    if s["pdc_settings"]["model_order"] == "auto":
        p_max = s["pdc_settings"]["max_order"]
        p, bic = compute_order(data, p_max=p_max)
    else:
        p = s["pdc_settings"]["model_order"]

    A_est, sigma = mvar_fit(data, p)
    sigma = diag(sigma)  # DTF + PDC support diagonal noise
    # sigma = None

    if s["pdc_settings"]["num_fft"] == "auto":
        n_fft = max(p, s["pdc_settings"]["max_order"]) + 1
    else:
        n_fft = int(s["pdc_settings"]["num_fft"])

    # compute DTF
    # D, freqs = DTF(A_est, sigma, n_fft)

    # compute PDC
    P, freqs = PDC(A_est, sigma, n_fft)
    P = nanmean(P, axis=0).squeeze()

    feature_name = '_'.join(["pdc", filt, "model_order"])
    features_[feature_name] = p

    for row in np_arange(P.shape[0]):
        for col in np_arange(P.shape[1]):
            feature_calc = P[row, col]
            ch_1 = chs[row]
            ch_2 = chs[col]
            feature_name = '_'.join(["pdc", filt, ch_1, "to", ch_2])
            features_[feature_name] = feature_calc
    return features_


def get_dtf(features_, s, data, filt, chs) -> dict:
    if s["dtf_settings"]["model_order"] == "auto":
        p_max = s["dtf_settings"]["max_order"]
        p, bic = compute_order(data, p_max=p_max)
    else:
        p = s["dtf_settings"]["model_order"]

    A_est, sigma = mvar_fit(data, p)
    sigma = diag(sigma)  # DTF + PDC support diagonal noise
    # sigma = None

    if s["dtf_settings"]["num_fft"] == "auto":
        n_fft = max(p, s["dtf_settings"]["max_order"]) + 1
    else:
        n_fft = int(s["dtf_settings"]["num_fft"])

    # compute DTF
    D, freqs = DTF(A_est, sigma, n_fft)
    D = nanmean(D, axis=0).squeeze()

    feature_name = '_'.join(["dtf", filt, "model_order"])
    features_[feature_name] = p

    for row in np_arange(D.shape[0]):
        for col in np_arange(D.shape[1]):
            feature_calc = D[row, col]
            ch_1 = chs[row]
            ch_2 = chs[col]
            feature_name = '_'.join(["dtf", filt, ch_1, "to", ch_2])
            features_[feature_name] = feature_calc
    return features_
