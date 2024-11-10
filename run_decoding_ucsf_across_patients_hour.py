import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight
#import cebra
#from cebra import CEBRA
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from xgboost import XGBClassifier
from matplotlib.backends.backend_pdf import PdfPages

PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

EXCLUDE_ZERO_UPDRS_DYK = False
PLT_ = False

subs_no_dyk = ["rcs10", "rcs14", "rcs15", "rcs19"]

MODEL_NAME = "CB" # "CB", "LM", "XGB", "PCA_LM", "CEBRA", "RF"

if __name__ == "__main__":

    df_all_ = pd.read_csv(os.path.join(PATH_READ, "all_merged_normed.csv"), index_col=0)

    subs = df_all_["sub"].unique()
    if EXCLUDE_ZERO_UPDRS_DYK:
        subs = [s for s in subs if not any([s_no for s_no in subs_no_dyk if s_no in s])]

    df_all_ = df_all_.drop(columns=df_all_.columns[df_all_.isnull().all()])
    df_all_["pkg_dt"] = pd.to_datetime(df_all_["pkg_dt"])

    for EXCLUDE_hour_feature in [True, False]:
        d_out = {}
        for CLASS in [True, False]:

            df_all = df_all_.copy()

            d_out[CLASS] = {}
            for label_idx, label_name in enumerate(["pkg_dk", "pkg_bk", "pkg_tremor"]): 

                d_out[CLASS][label_name] = {}

                mask = ~df_all[label_name].isnull()
                df_all = df_all[mask].copy()
                
                for loc_ in ["ecog_stn", ]:  # "ecog", "stn"
                    #if loc_ != "ecog":
                    #    continue
                    d_out[CLASS][label_name][loc_] = {}
                    if PLT_:
                        pdf_pages = PdfPages(os.path.join(PATH_FIGURES, f"decoding_across_patients_class_{CLASS}_{loc_}_10s_segmentlength_all_{MODEL_NAME}_tests.pdf")) 
                    if loc_ == "ecog_stn":
                        df_use = df_all.copy()
                    elif loc_ == "ecog":
                        df_use = df_all[[c for c in df_all.columns if c.startswith("cortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
                    elif loc_ == "stn":
                        df_use = df_all[[c for c in df_all.columns if c.startswith("ch_subcortex") or c.startswith("pkg") or c.startswith("sub")]].copy()

                    if CLASS and "_dk" in label_name:
                        df_use[label_name] = (df_use[label_name].copy() / df_use[label_name].max()) > 0.02
                    elif CLASS and "_bk" in label_name:
                        df_use[label_name] = df_use[label_name].copy() > 50
                    elif CLASS and "_tremor" in label_name:
                        df_use[label_name] = df_use[label_name].copy() > 1

                    for sub_test in subs:  # tqdm(
                        print(f"sub_test: {sub_test}")

                        df_test = df_use[df_use["sub"] == sub_test]
                        df_test = df_test.drop(columns=["sub"])
                        y_test = np.array(df_test[label_name])
                        df_train = df_use[df_use["sub"] != sub_test]
                        df_train = df_train.drop(columns=["sub"])
                        y_train = np.array(df_train[label_name])
                        X_train = df_train[[c for c in df_train.columns if "pkg" not in c]]
                        if EXCLUDE_hour_feature is False:
                            X_train["hour"] = df_train["pkg_dt"].dt.hour

                        X_test = df_test[[c for c in df_test.columns if "pkg" not in c]]
                        if EXCLUDE_hour_feature is False:
                            X_test["hour"] = df_test["pkg_dt"].dt.hour
                        
                        if CLASS:
                            classes = np.unique(y_train)
                            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                            class_weights = dict(zip(classes, weights))
                            model = CatBoostClassifier(silent=True, class_weights=class_weights)
                        else:
                            model = CatBoostRegressor(silent=True) # task_type="GPU"

                        # drop columns that have NaN values
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


                        model.fit(X_train, y_train)
                        pr = model.predict(X_test)
                        feature_importances = model.get_feature_importance(Pool(X_test, y_test), type="PredictionValuesChange")


                        d_out[CLASS][label_name][loc_][sub_test] = {}
                        if CLASS:
                            y_test = y_test.astype(int)
                            pr = pr.astype(int)
                            d_out[CLASS][label_name][loc_][sub_test]["accuracy"] = metrics.accuracy_score(y_test, pr)
                            d_out[CLASS][label_name][loc_][sub_test]["ba"] = metrics.balanced_accuracy_score(y_test, pr)
                            d_out[CLASS][label_name][loc_][sub_test]["f1"] = metrics.f1_score(y_test, pr)
                            d_out[CLASS][label_name][loc_][sub_test]["pr_proba"] = model.predict_proba(X_test)
                        else:
                            corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
                            d_out[CLASS][label_name][loc_][sub_test]["corr_coeff"] = corr_coeff
                            d_out[CLASS][label_name][loc_][sub_test]["r2"] = metrics.r2_score(y_test, pr)
                            d_out[CLASS][label_name][loc_][sub_test]["mse"] = metrics.mean_squared_error(y_test, pr)
                            d_out[CLASS][label_name][loc_][sub_test]["mae"] = metrics.mean_absolute_error(y_test, pr)
                        d_out[CLASS][label_name][loc_][sub_test]["pr"] = pr
                        d_out[CLASS][label_name][loc_][sub_test]["y_"] = y_test
                        d_out[CLASS][label_name][loc_][sub_test]["time"] = df_test["pkg_dt"].values
                        d_out[CLASS][label_name][loc_][sub_test]["feature_importances"] = feature_importances

        SAVE_NAME = f"LOHO_ALL_LABELS_ALL_GROUPS_exludehour_{EXCLUDE_hour_feature}.pkl"
        with open(os.path.join(PATH_PER, SAVE_NAME), "wb") as f:
            pickle.dump(d_out, f)
