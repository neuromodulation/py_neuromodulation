import pickle
import os
import numpy as np
from sklearn import linear_model

from py_neuromodulation import nm_analysis

PATH_OUT = r"C:\Users\ICN_admin\Documents\LSL_Test"
folder_name = "Test"


if __name__ == "__main__":

    analyzer = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=folder_name, binarize_label=False
    )
    X = analyzer.feature_arr.iloc[:, :-1]
    y = analyzer.feature_arr.iloc[:, -1]
    model = linear_model.LogisticRegression().fit(X, y)

    with open(
        os.path.join(PATH_OUT, folder_name, "linear_model.pkl"), "wb"
    ) as fid:
        pickle.dump(model, fid)
