import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_plots,
    nm_stats,
)
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

PATH_OUT = "/home/timonmerk/Documents/PN_OUT"
sub_name = "PIT-RNS0427"

# init analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=sub_name
)

arr = stats.zscore(feature_reader.feature_arr)
arr_background = arr[arr["ch1_LineLength"]<np.percentile(arr["ch1_LineLength"], q=75)]

plt.imshow(arr_background.T, aspect="auto")
plt.yticks(np.arange(arr_background.shape[1]), feature_reader.feature_arr.columns)
plt.show()

plt.imshow(arr.T, aspect="auto")
plt.yticks(np.arange(arr.shape[1]), feature_reader.feature_arr.columns)
plt.colorbar()
plt.clim(-1, 1)
plt.show()