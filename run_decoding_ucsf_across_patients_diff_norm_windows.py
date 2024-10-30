import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized"
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
PLT_PDF = False

if __name__ == "__main__":

    for label_name in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
        if label_name == "pkg_bk":
            CLASSIFICATION = False
        else:
            CLASSIFICATION = True

        for norm_window in [0, 5, 10, 20, 30, 60, 120, 180, 300, 480, 720, 960, 1200, 1440]:
            
            # check if pickel output file exists
            if  os.path.exists(os.path.join(PATH_PER, f"d_out_patient_across_{label_name}_class_{CLASSIFICATION}_{str(norm_window)}.pkl")):
                continue
            
            if norm_window == 0:
                PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length"
                df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged_preprocessed.csv"), index_col=0)
            else:
                PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length"
                df_all = pd.read_csv(os.path.join(PATH_OUT, str(norm_window), "all_merged_normed.csv"), index_col=0)
            #df_all = df_all.drop(columns=["Unnamed: 0"])
            subs = df_all["sub"].unique()

            d_out = {}
            print(label_name)
            d_out[label_name] = {}

            df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
            df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
            mask = ~df_all[label_name].isnull()
            df_all = df_all[mask]
            
            #for loc_ in ["ecog_stn", "ecog", "stn",]:
            loc_ = "ecog_stn"
            d_out[label_name][loc_] = {}
            if PLT_PDF:
                pdf_pages = PdfPages(os.path.join("figures_ucsf", f"decoding_across_patients_class_{label_name}_{loc_}.pdf")) 
            if loc_ == "ecog_stn":
                df_use = df_all.copy()
            elif loc_ == "ecog":
                df_use = df_all[[c for c in df_all.columns if c.startswith("ch_cortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
            elif loc_ == "stn":
                df_use = df_all[[c for c in df_all.columns if c.startswith("ch_subcortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
            
            if "_dk" in label_name:
                df_use[label_name] = (df_use[label_name].copy() / df_use[label_name].max()) > 0.02
            elif "_tremor" in label_name:
                df_use[label_name] = df_use[label_name].copy() > 1
                
            for sub_test in tqdm(subs):
                df_test = df_use[df_use["sub"] == sub_test]

                df_test = df_test.drop(columns=["sub"])
                y_test = np.array(df_test[label_name])

                df_train = df_use[df_use["sub"] != sub_test]
                df_train = df_train.drop(columns=["sub"])
                y_train = np.array(df_train[label_name])

                X_train = df_train[[c for c in df_train.columns if "pkg" not in c]]
                X_train["hour"] = df_train["pkg_dt"].dt.hour

                X_test = df_test[[c for c in df_test.columns if "pkg" not in c]]
                X_test["hour"] = df_test["pkg_dt"].dt.hour
                
                #X_ = X.dropna(axis=1)  # drop all columns that have NaN values
                if CLASSIFICATION:
                    classes = np.unique(y_train)
                    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    class_weights = dict(zip(classes, weights))
                    model = CatBoostClassifier(silent=True, class_weights=class_weights)
                    #model = linear_model.LogisticRegression()
                else:
                    model = CatBoostRegressor(silent=True) # task_type="GPU"

                model.fit(X_train, y_train)

                pr = model.predict(X_test)

                feature_importances = model.get_feature_importance(Pool(X_test, y_test), type="PredictionValuesChange")

                d_out[label_name][loc_][sub_test] = {}
                if CLASSIFICATION:
                    y_test = y_test.astype(int)
                    pr = pr.astype(int)
                    d_out[label_name][loc_][sub_test]["accuracy"] = metrics.accuracy_score(y_test, pr)
                    d_out[label_name][loc_][sub_test]["ba"] = metrics.balanced_accuracy_score(y_test, pr)
                    d_out[label_name][loc_][sub_test]["f1"] = metrics.f1_score(y_test, pr)
                    d_out[label_name][loc_][sub_test]["pr_proba"] = model.predict_proba(X_test)
                else:
                    corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
                    d_out[label_name][loc_][sub_test]["corr_coeff"] = corr_coeff
                    d_out[label_name][loc_][sub_test]["r2"] = metrics.r2_score(y_test, pr)
                    d_out[label_name][loc_][sub_test]["mse"] = metrics.mean_squared_error(y_test, pr)
                    d_out[label_name][loc_][sub_test]["mae"] = metrics.mean_absolute_error(y_test, pr)
                d_out[label_name][loc_][sub_test]["pr"] = pr
                d_out[label_name][loc_][sub_test]["y_"] = y_test
                d_out[label_name][loc_][sub_test]["time"] = df_test["pkg_dt"].values
                d_out[label_name][loc_][sub_test]["feature_importances"] = feature_importances

                if PLT_PDF:
                    plt.figure(figsize=(10, 4), dpi=200)
                    #plt.plot(y_test, label="true")
                    #plt.plot(pr, label="pr")
                    plt.plot(d_out[label_name][loc_][sub_test]["pr_proba"][:, 1], label="pr_proba")
                    plt.plot(d_out[label_name][loc_][sub_test]["true_reg_normed"].values, label="true")

                    plt.legend()
                    plt.ylabel(f"PKG score {label_name}")
                    plt.xlabel("Time [30s]")
                    if CLASSIFICATION:
                        plt.title(f"ba: {np.round(d_out[label_name][loc_][sub_test]['ba'], 2)} sub: {sub_test}")
                    else:
                        plt.title(f"corr_coeff: {np.round(corr_coeff, 2)} sub: {sub_test}")
                    pdf_pages.savefig(plt.gcf())
                    plt.close()
                if PLT_PDF:
                    pdf_pages.close()


            SAVE_NAME = f"d_out_patient_across_{label_name}_class_{CLASSIFICATION}_{str(norm_window)}.pkl"

            with open(os.path.join(PATH_PER, SAVE_NAME), "wb") as f:
                pickle.dump(d_out, f)
