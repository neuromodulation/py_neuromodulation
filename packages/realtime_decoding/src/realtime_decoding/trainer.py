"""
Read saved features from timeflux .hdf5 file and train model
"""
from sklearn import linear_model, model_selection, metrics
import pickle
import pandas as pd
import os
import numpy as np

#from TMSiFileFormats.file_readers import Poly5Reader

#from matplotlib import pyplot as plt

if __name__ == "__main__":

    # read data for rotameter amplitude investigation:

    #data = Poly5Reader(r"C:\CODE\py_neuromodulation\realtime_experiment\data\testsub\EcogLfpMedOff01\sub_ses_task_acq_run_datatype-20230320_145212.poly5")

    sub = "559NB58"
    run = 1
    PATH_HDF5_FEATURES = rf"C:\CODE\py_neuromodulation\realtime_experiment\data\{sub}\EcogLfpMedOn01\{sub}_EcogLfpMedOn01_SelfpRotaL_StimOff_{str(run)}_ieeg.hdf5"

    PATH_MODEL_SAVE = os.path.join(
        rf"C:\CODE\py_neuromodulation\realtime_experiment\data\{sub}\EcogLfpMedOn01",
        "model_trained.p"
    )

    df = pd.read_hdf(PATH_HDF5_FEATURES, key="features")

    y = np.abs(np.diff(df["ROTAMETER_MOVEMENT"]))
    X = df[[f for f in df.columns if "time" not in f and "ROTAMETER_MOVEMENT" not in f]].iloc[1:, :]


    # put here in debug line
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(np.array(df["ROTAMETER_MOVEMENT"]))
    plt.plot(y)
    plt.show()

    LIM_ = 200
    X_lim = X.iloc[LIM_:, :]
    y_lim = y[LIM_:]


    model = linear_model.LogisticRegression()

    #model = model.fit(X, y>0.01)
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