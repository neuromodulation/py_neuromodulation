"""
Read saved features from timeflux .hdf5 file and train model
"""
from sklearn import linear_model
import pickle
import pandas as pd


if __name__ == "__main__":
    PATH_HDF5_FEATURES = "/here.hdf5"
    PATH_MODEL_SAVE = "model_trained.p"
    df = pd.read_hdf(PATH_HDF5_FEATURES)

    y = df["label"]
    X = df[[f for f in df.columns if "time" not in f and "label" not in f]]
    model = linear_model.LogisticRegression().fit(X, y)

    with open(PATH_MODEL_SAVE, "wb") as fid:
        pickle.dump(model, fid)
