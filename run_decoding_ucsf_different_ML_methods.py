import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight

import cebra
from cebra import CEBRA
from tqdm import tqdm
from scipy import stats
#from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor

PLT_ = False
loc_ = "ecog_stn"

if __name__ == "__main__":
    PATH_DATA = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480/all_merged_normed.csv"
    PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
    df_orig = pd.read_csv(PATH_DATA, index_col=0)

    for MODEL_NAME in ["XGB"]: #["CB", "RF",]:  # "XGB", "CEBRA", "LM", "PCA_LM",
        for label_name in ["pkg_dk", "pkg_tremor", "pkg_bk", ]:
            # check if outfile exists
            if os.path.exists(
                os.path.join(
                    PATH_OUT,
                    f"d_out_ML_across_patients_{label_name}_10s_seglength_480_all_{MODEL_NAME}.pkl",
                )
            ):
                print()
            if label_name == "pkg_bk":
                CLASSIFICATION = False
            else:
                CLASSIFICATION = True

            df_all = df_orig.copy() #[[c for c in df_orig.columns if "pkg_" in c or c == "sub"]].copy()

            subs = df_all["sub"].unique()

            d_out = {}

            df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
            df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])

            d_out = {}
            
            if loc_ == "ecog_stn":
                df_use = df_all.copy()

            if "_dk" in label_name:
                df_use[label_name] = (df_use[label_name].copy() / df_use[label_name].max()) > 0.02
            elif "_tremor" in label_name:
                df_use[label_name] = df_use[label_name].copy() > 1

            for sub_test in tqdm(subs):  # tqdm(
                d_out[sub_test] = {}
                df_test = df_use[df_use["sub"] == sub_test].copy()

                df_test = df_test.drop(columns=["sub"])
                # y_test = np.array(df_test[label_name])
                y_test = np.array(df_test[label_name])

                df_train = df_use[df_use["sub"] != sub_test].copy()
                df_train = df_train.drop(columns=["sub"])
                y_train = np.array(df_train[label_name])

                X_train = df_train[
                    [c for c in df_train.columns if "pkg" not in c]
                ]  #  and "psd" not in c
                X_train["hour"] = df_train["pkg_dt"].dt.hour

                X_test = df_test[
                    [c for c in df_test.columns if "pkg" not in c]
                ]  #  and "psd" not in c
                X_test["hour"] = df_test["pkg_dt"].dt.hour

                # X_ = X.dropna(axis=1)  # drop all columns that have NaN values
                if CLASSIFICATION:
                    classes = np.unique(y_train)
                    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    class_weights = dict(zip(classes, weights))
                    if MODEL_NAME == "CB":
                        model = CatBoostClassifier(
                            silent=True,
                            #loss_function="MultiLogloss",
                            #task_type="GPU",
                            #devices="0",
                            class_weights=class_weights,
                        )  # class_weights=class_weights
                    elif MODEL_NAME == "XGB":
                        model = XGBClassifier(class_weight="balanced", n_jobs=-1, verbose=True)
                    elif MODEL_NAME == "RF":
                        model = ensemble.RandomForestClassifier(
                            class_weight="balanced", n_jobs=-1
                        )
                    elif MODEL_NAME == "LM":
                        model = linear_model.LogisticRegression(class_weight="balanced")
                    
                else:
                    if MODEL_NAME == "CB":
                        model = CatBoostRegressor(silent=True)
                    elif MODEL_NAME == "XGB":
                        model = XGBRegressor(n_jobs=-1, verbose=True)
                    elif MODEL_NAME == "RF":
                        model = ensemble.RandomForestRegressor(n_jobs=-1)
                    elif MODEL_NAME == "LM":
                        model = linear_model.LinearRegression()

                # drop columns that have NaN values
                if MODEL_NAME != "CB":
                    X_train = X_train.dropna(axis=1)
                    X_test = X_test[X_train.columns]

                    # drop columns that have inf values
                    X_train = X_train.replace([np.inf, -np.inf], np.nan)
                    X_train = X_train.dropna(axis=1)
                    X_test = X_test[X_train.columns]

                    # drop X_test columns that have inf values
                    X_test = X_test.replace([np.inf, -np.inf], np.nan)
                    X_test = X_test.dropna(axis=1)
                    X_train = X_train[X_test.columns]

                    # which columns contain inf
                    # replace NaN values with 0
                    X_test = X_test.fillna(0)

                if MODEL_NAME == "PCA_LM":
                    pca = PCA(n_components=10)
                    # standardize the data
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                    if CLASSIFICATION:
                        model = linear_model.LogisticRegression(class_weight="balanced")
                    else:
                        model = linear_model.LinearRegression()
                elif MODEL_NAME == "CEBRA":
                    cebra_model = CEBRA(
                        model_architecture="offset10-model",  #'offset40-model-4x-subsample', # previously used: offset1-model-v2'    # offset10-model  # my-model
                        batch_size=100,
                        temperature_mode="auto",
                        learning_rate=0.005,
                        max_iterations=2000,
                        # time_offsets = 10,
                        output_dimension=3,  # check 10 for better performance
                        device="mps",
                        # conditional="time_delta",  # assigning CEBRA to sample temporally and behaviorally for reference
                        hybrid=False,
                        verbose=True,
                    )

                    cebra_model.fit(X_train, y_train)
                    X_train = cebra_model.transform(X_train)
                    X_test = cebra_model.transform(X_test)
                    # cebra.plot_loss(cebra_model)
                    # cebra.plot_temperature(cebra_model)
                    # cebra.plot_embedding(
                    #     X_train,
                    #     cmap="viridis",
                    #     markersize=10,
                    #     alpha=0.5,
                    #     embedding_labels=y_train, #np.clip(y_train, 0, 80),
                    # )  # embedding_labels=y_train
                    # plt.show(block=True)
                    if CLASSIFICATION:
                        model = linear_model.LogisticRegression(class_weight="balanced")
                    else:
                        model = linear_model.LinearRegression()

                #arr = np.random.random([100, 10])
                #label_ = np.random.random([100, 1])
                # uv pip 1.7.6 works
                model.fit(X_train, y_train, verbose=True)

                pr = model.predict(X_test)
                feature_importances = None
                if type(model) == linear_model.LogisticRegression or type(model) == linear_model.LinearRegression:
                    feature_importances = model.coef_
                elif type(model) == XGBClassifier or type(model) == XGBRegressor:
                    feature_importances = model.feature_importances_
                elif MODEL_NAME == "CB":
                    feature_importances = model.get_feature_importance(
                        Pool(X_test, y_test), type="PredictionValuesChange"
                    )
                elif MODEL_NAME == "RF":
                    feature_importances = model.feature_importances_

                if CLASSIFICATION:
                    y_test = y_test.astype(int)
                    pr = pr.astype(int)
                    d_out[sub_test]["accuracy"] = metrics.accuracy_score(
                        y_test, pr
                    )
                    d_out[sub_test]["pr_proba"] = model.predict_proba(
                        X_test
                    )
                    d_out[sub_test]["ba"] = metrics.balanced_accuracy_score(
                        y_test, pr
                    )

                else:
                    corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
                    d_out[sub_test]["corr_coeff"] = corr_coeff
                    d_out[sub_test]["r2"] = metrics.r2_score(y_test, pr)
                    d_out[sub_test]["mse"] = metrics.mean_squared_error(
                        y_test, pr
                    )
                    d_out[sub_test]["mae"] = metrics.mean_absolute_error(
                        y_test, pr
                    )
                d_out[sub_test]["pr"] = pr
                d_out[sub_test]["y_"] = y_test
                d_out[sub_test]["time"] = df_test["pkg_dt"].values
                d_out[sub_test]["feature_importances"] = feature_importances

            SAVE_NAME = f"d_out_ML_across_patients_{label_name}_10s_seglength_480_all_{MODEL_NAME}.pkl"

            with open(os.path.join(PATH_OUT, SAVE_NAME), "wb") as f:
                pickle.dump(d_out, f)
