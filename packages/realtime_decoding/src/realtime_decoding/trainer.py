"""
Read saved features from timeflux .hdf5 file and train model
"""
from sklearn import linear_model
import pickle
import pandas as pd
import os


if __name__ == "__main__":
    PATH_HDF5_FEATURES = r"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-937\ses-EcogLfpMedOff01\sub-937_ses-EcogLfpMedOff01_task-RealtimeDecodingR_acq-StimOff_run-1_ieeg.hdf5"
    PATH_MODEL_SAVE = os.path.join(
        r"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-937\ses-EcogLfpMedOff01",
        "model_trained.p"
    )
    df = pd.read_hdf(PATH_HDF5_FEATURES, key="features")

    y = df["label_train"]
    X = df[[f for f in df.columns if "time" not in f and "label" not in f]]
    model = linear_model.LogisticRegression().fit(X, y)

    with open(PATH_MODEL_SAVE, "wb") as fid:
        pickle.dump(model, fid)
