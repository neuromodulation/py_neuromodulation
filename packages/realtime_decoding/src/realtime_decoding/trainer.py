"""
Read saved features from timeflux .hdf5 file and train model
"""
from sklearn import linear_model
import pickle
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":

    sub = 937

    PATH_HDF5_FEATURES = rf"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-{sub}\ses-EcogLfpMedOff01\sub-{sub}_ses-EcogLfpMedOff01_task-RealtimeDecodingR_acq-StimOff_run-1_ieeg.hdf5"
    PATH_MODEL_SAVE = os.path.join(
        rf"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-{sub}\ses-EcogLfpMedOff01",
        "model_trained.p"
    )

    df = pd.read_hdf(PATH_HDF5_FEATURES, key="features")

    y = np.abs(np.diff(df["label_train"]))
    X = df[[f for f in df.columns if "time" not in f and "label" not in f]]

    model = linear_model.LogisticRegression()

    model = model.fit(X, y)

    with open(PATH_MODEL_SAVE, "wb") as fid:
        pickle.dump(model, fid)
