"""
Read saved features from timeflux .hdf5 file and train model
"""
from sklearn import linear_model, model_selection, metrics
import pickle
import pandas as pd
import os
import numpy as np
#from matplotlib import pyplot as plt

if __name__ == "__main__":

    sub = "654"

    PATH_HDF5_FEATURES = rf"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-{sub}\ses-EcogLfpMedOff01\sub-{sub}_ses-EcogLfpMedOff01_task-RealtimeDecodingR_acq-StimOff_run-1_ieeg.hdf5"
    PATH_MODEL_SAVE = os.path.join(
        rf"C:\CODE\py_neuromodulation\realtime_experiment\data\sub-{sub}\ses-EcogLfpMedOff01",
        "model_trained.p"
    )

    df = pd.read_hdf(PATH_HDF5_FEATURES, key="features")

    y = np.abs(np.diff(df["label_train"]))
    X = df[[f for f in df.columns if "time" not in f and "label" not in f]].iloc[1:, :]

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.plot(np.array(df["label_train"]))
    #plt.plot(y)
    #plt.show()

    X_lim = X.iloc[570:, :]
    y_lim = y[570:]


    model = linear_model.LogisticRegression()

    model = model.fit(X, y>0.01)
    model = model.fit(X_lim, y_lim>0.01)

    with open(PATH_MODEL_SAVE, "wb") as fid:
        pickle.dump(model, fid)

    pr = model_selection.cross_val_predict(
        estimator=linear_model.LinearRegression(),
        X=X_lim,
        y=y_lim>0.01,
        cv=model_selection.KFold(n_splits=3, shuffle=False)
    )

    plt.figure()
    plt.plot(pr)
    plt.plot(y_lim)
    plt.show()