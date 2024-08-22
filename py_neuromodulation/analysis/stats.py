import random
import copy

import matplotlib.pyplot as plt

# from numba import njit
import numpy as np
import pandas as pd
import scipy.stats as stats


def fitlm_kfold(x, y, kfold_splits=5):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    model = LinearRegression()
    if isinstance(x, type(np.array([]))) or isinstance(x, type([])):
        x = pd.DataFrame(x)
    if isinstance(y, type(np.array([]))) or isinstance(y, type([])):
        y = pd.DataFrame(y)
    scores, coeffs = [], np.zeros(x.shape[1])
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(x, y)):
        model.fit(x.iloc[train, :], y.iloc[train, :])
        score = model.score(x.iloc[test, :], y.iloc[test, :])
        scores.append(score)
        coeffs = np.vstack((coeffs, model.coef_))
    coeffs = list(np.delete(coeffs, 0))
    return scores, coeffs, model, ["scores", "coeffs", "model"]


def zscore(data):
    return (data - data.mean()) / data.std()


def permutationTestSpearmansRho(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test for multiple repetitions of Spearmans Rho
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distibution e.g. R^2
    y (np array) : second distribution e.g. UPDRS
    plot_distr (boolean) : if True: permutation histplot and ground truth will be
    plotted
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here spearman's rho
    p (float) : p value of permutation test
    """

    # compute ground truth difference
    gT = stats.spearmanr(x, y)[0]
    #
    pV = np.array((x, y))
    # Initialize permutation:
    pD = []
    # Permutation loop:
    args_order = np.arange(0, pV.shape[1], 1)
    args_order_2 = np.arange(0, pV.shape[1], 1)
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(args_order)
        random.shuffle(args_order_2)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(stats.spearmanr(pV[0, args_order], pV[1, args_order_2])[0])

    # calculate p value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val


def permutationTest(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    # Compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))

    pV = np.concatenate((x, y), axis=0)
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(
            np.abs(
                np.average(pS[0 : int(len(pS) / 2)])
                - np.average(pS[int(len(pS) / 2) :])
            )
        )

    # Calculate p-value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val


def permutationTest_relative(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    gT = np.abs(np.average(x) - np.average(y))
    pD = []
    for i in range(0, p):
        l_ = []
        for i in range(x.shape[0]):
            if random.randint(0, 1) == 1:
                l_.append((x[i], y[i]))
            else:
                l_.append((y[i], x[i]))
        pD.append(
            np.abs(np.average(np.array(l_)[:, 0]) - np.average(np.array(l_)[:, 1]))
        )
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()

    return gT, p_val


# @njit
def permutation_numba_onesample(x, y, n_perm, two_tailed=True):
    """Perform permutation test with one-sample distribution.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int or float
        Baseline against which to check for statistical significane
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """
    if two_tailed:
        zeroed = x - y
        z = np.abs(np.mean(zeroed))
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(a=np.array([-1.0, 1.0]), size=len(x), replace=True)
            p[i] = np.abs(np.mean(zeroed * sign))
    else:
        zeroed = x - y
        z = np.mean(zeroed)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(a=np.array([-1.0, 1.0]), size=len(x), replace=True)
            p[i] = np.mean(zeroed * sign)
        # Return p-value
    return z, (np.sum(p >= z)) / n_perm


# @njit
def permutation_numba_twosample(x, y, n_perm, two_tailed=True):
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like
        Second distribution
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means
    float
        P-value of permutation test
    """
    if two_tailed:
        z = np.abs(np.mean(x) - np.mean(y))
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
    else:
        z = np.mean(x) - np.mean(y)
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(pS[:half]) - np.mean(pS[half:])
    return z, (np.sum(p >= z)) / n_perm


def cluster_wise_p_val_correction(p_arr, p_sig=0.05, num_permutations=10000):
    """Obtain cluster-wise corrected p values.

    Based on: https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_arr (np.array) : ndim, can be time series or image
    p_sig (float) : significance level
    num_permutations (int) : no. of random permutations of cluster comparisons

    Returns
    -------
    p (float) : significance level of highest cluster
    p_min_index : indices of significant samples
    """
    from skimage.measure import label as measure_label

    labels, num_clusters = measure_label(p_arr <= p_sig, return_num=True)

    # loop through clusters of p_val series or image
    index_cluster = {}
    p_cluster_sum = np.zeros(num_clusters)
    for cluster_i in np.arange(num_clusters):
        # first cluster is assigned to be 1 from measure.label
        index_cluster[cluster_i] = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum[cluster_i] = np.sum(np.array(1 - p_arr)[index_cluster[cluster_i]])
    # p_min corresponds to the most unlikely cluster
    p_min = np.max(p_cluster_sum)

    p_min_index = index_cluster[np.argmax(p_cluster_sum)]

    # loop through random permutation cycles
    r_per_arr = np.zeros(num_permutations)
    for r in range(num_permutations):
        r_per = np.random.randint(low=0, high=p_arr.shape[0], size=p_arr.shape[0])

        labels, num_clusters = measure_label(p_arr[r_per] <= p_sig, return_num=True)

        index_cluster = {}
        if num_clusters == 0:
            r_per_arr[r] = 0
        else:
            p_cluster_sum = np.zeros(num_clusters)
            for cluster_i in np.arange(num_clusters):
                index_cluster[cluster_i] = np.where(labels == cluster_i + 1)[
                    0
                ]  # first cluster is assigned to be 1 from measure.label
                p_cluster_sum[cluster_i] = np.sum(
                    np.array(1 - p_arr[r_per])[index_cluster[cluster_i]]
                )
            # corresponds to the most unlikely cluster
            r_per_arr[r] = np.max(p_cluster_sum)

    sorted_r = np.sort(r_per_arr)

    def find_arg_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    p = 1 - find_arg_nearest(sorted_r, p_min) / num_permutations

    return p, p_min_index


# @njit
def cluster_wise_p_val_correction_numba(p_arr, p_sig, n_perm):
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_arr :  array-like
        Array of p-values. WARNING: MUST be one-dimensional
    p_sig : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution

    Returns
    -------
    p : list of floats
        List of p-values for each cluster
    p_min_index : list of numpy array
        List of indices of each significant cluster
    """

    def cluster(iterable):
        """Cluster 1-D array of boolean values.

        Parameters
        ----------
        iterable : array-like of bool
            Array to be clustered.

        Returns
        -------
        cluster_labels : np.ndarray
            Array of shape (len(iterable), 1), where each value indicates the
            number of the cluster. Values are 0 if the item does not belong to
            a cluster
        cluster_count : int
            Number of detected cluster. Corresponds to the highest value in
            cluster_labels
        """
        cluster_labels = np.zeros((len(iterable), 1))
        cluster_count = 0
        cluster_len = 0
        for idx, item in enumerate(iterable):
            if item:
                cluster_labels[idx] = cluster_count + 1
                cluster_len += 1
            elif cluster_len == 0:
                pass
            else:
                cluster_len = 0
                cluster_count += 1
        if cluster_len >= 1:
            cluster_count += 1
        return cluster_labels, cluster_count

    def calculate_null_distribution(p_arr_, p_sig_, n_perm_):
        """Calculate null distribution of clusters.

        Parameters
        ----------
        p_arr_ :  numpy array
            Array of p-values
        p_sig_ : float
            Significance level (p-value)
        n_perm_ : int
            No. of random permutations

        Returns
        -------
        r_per_arr : numpy array
            Null distribution of shape (n_perm_)
        """
        # loop through random permutation cycles
        r_per_arr = np.zeros(n_perm_)
        for r in range(n_perm_):
            r_per = np.random.randint(low=0, high=p_arr_.shape[0], size=p_arr_.shape[0])
            labels_, n_clusters = cluster(p_arr_[r_per] <= p_sig_)

            cluster_ind = {}
            if n_clusters == 0:
                r_per_arr[r] = 0
            else:
                p_sum = np.zeros(n_clusters)
                for ind in range(n_clusters):
                    cluster_ind[ind] = np.where(labels_ == ind + 1)[0]
                    p_sum[ind] = np.sum(np.asarray(1 - p_arr_[r_per])[cluster_ind[ind]])
                r_per_arr[r] = np.max(p_sum)
        return r_per_arr

    labels, num_clusters = cluster(p_arr <= p_sig)

    null_distr = calculate_null_distribution(p_arr, p_sig, n_perm)
    # Loop through clusters of p_val series or image
    clusters = []
    p_vals = [np.float64(x) for x in range(0)]
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        index_cluster = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum = np.sum(np.asarray(1 - p_arr)[index_cluster])
        p_val = 1 - np.sum(p_cluster_sum >= null_distr) / n_perm
        if p_val <= p_sig:
            clusters.append(index_cluster)
            p_vals.append(p_val)

    return p_vals, clusters
