import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
import numpy as np
from sklearn import metrics, model_selection, linear_model
from joblib import Parallel, delayed
from catboost import CatBoostClassifier
from tqdm import tqdm

PATH_FEATURES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708\features_all_with_lfa.csv"
PATH_FEATURES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\0210\raw_new\features_all_with_lfa.csv"
PATH_FEATURES = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/EyesOpenBeijing/0210/raw_new/features_all_with_lfa.csv'

def balance_classes(df, target_column):
    min_class_size = df[target_column].value_counts().min()
    balanced_df = pd.concat([
        resample(df[df[target_column] == cls], 
                 replace=False, 
                 n_samples=min_class_size, 
                 random_state=42)
        for cls in df[target_column].unique()
    ])
    return balanced_df

def run_cv(df):
    # run a three fold cross validation to decode the label column, remove the time column
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    #model = linear_model.LinearRegression()
    cm_list = []
    ba_list = []
    coef_list = []
    for train_, test_ in cv.split(df, df["label_enc"]):
        # integer encode the label column
        train_df = df.iloc[train_]
        train_df_balanced = balance_classes(train_df, "label_enc")

        X_train = train_df_balanced.drop(columns=["label_enc"])
        y_train = train_df_balanced["label_enc"]

        # ensure that the same number of samples are taken for training from each class
        X_test = df.iloc[test_, :]
        y_test = df.iloc[test_, :]["label_enc"]
        X_test = X_test.drop(columns=["label_enc"])
        
        # include only columns that where all feature_list elements needs to be in the channel name
        #cols_use = [c for c in X_train.columns if all([f in c for f in features])]
        #X_train = X_train[cols_use]
        #X_test = X_test[cols_use]

        model = linear_model.LogisticRegression(class_weight="balanced")
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        
        # plot the confusion matrix
        cm = metrics.confusion_matrix(y_test, pred, ) # normalize="false"
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["EyesOpen", "EyesClosed"])
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix")
        # plt.show(block=True)

        cm_list.append(cm)
        ba_list.append(metrics.balanced_accuracy_score(y_test, pred))
        if type(model) == linear_model.LogisticRegression:
            coef_list.append(model.coef_)
        else:
            coef_list.append(model.get_feature_importance())
        print(metrics.balanced_accuracy_score(y_test, pred))
    
    # plot the mean confusion matrix
    cm_mean = sum(cm_list) / len(cm_list)
    #disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=["SLEEP", "EyesOpen", "EyesClosed"])
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Mean Confusion Matrix")
    #plt.show(block=True)

    ba_mean = sum(ba_list) / len(ba_list)
    return ba_mean, cm_mean, np.mean(coef_list, axis=0)

def compute_modality(sub, mod):
    
    #for mod in modality_:
    l_df = []
    if mod != "all":
        df_mod = df_all[[c for c in df_all.columns if mod in c] + ["label_enc"] + ["sub"] + ["disease"]].copy()
    else:
        df_mod = df_all.copy()

    df_sub = df_mod[df_mod["sub"] == sub]
    disease = df_sub["disease"].unique()[0]
    for loc in locs:
        print(loc)
        cols_loc = [c for c in df_sub.columns if loc in c] + ["label_enc"]
        if len(cols_loc) == 0:
            continue
        df_sub_loc = df_sub[cols_loc].copy()
        # select only columns that have non NaN values
        df_sub_loc = df_sub_loc.dropna(axis=1)
        if len(df_sub_loc.columns) == 1:
            continue
        
        chs_ = np.unique([c.split("_")[0] for c in df_sub_loc.columns if "label" not in c])
        for ch in chs_:
            print(ch)
            #ch = 'RSTN3-RSTN4'
            df_sub_ch = df_sub_loc[[c for c in df_sub_loc.columns if ch in c or "label" in c]].copy()
            f_bands = [c.split("_")[2] for c in df_sub_ch.columns if "label" not in c]
            ba_mean, cm_mean, coef_mean = run_cv(df_sub_ch.copy())
        
            dict_out = {"sub": sub, "loc": loc, "ch" : str(ch), "mod": mod, "dout": disease, "ba": float(ba_mean)}
            #for i, coef in enumerate(coef_mean.T):
            #    dict_out[f"coef_{f_bands[i]}"] = float(coef)
            l_df.append(dict_out)
    return l_df

if __name__ == "__main__":

    df_all = pd.read_csv(PATH_FEATURES)
    df_all["label_enc"] = df_all["label"].map({"SLEEP": 0, "EyesOpen": 1, "EyesClosed": 2})
    

    DECODE_SLEEP = False
    if not DECODE_SLEEP:
        df_all.query("label_enc != 0", inplace=True)

    df_all = df_all.drop(columns=["time", "label"])
    np.random.seed()

    locs = ["STN", "GPI"]  #  "ECOG", "EEG"
    #modality_ = ["theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA", "fft", "fooof", "all"]
    modality_ = ["fft", "alpha"]  # "low_frequency_activity", "alpha", 

    subs = df_all["sub"].unique()
    diseases = df_all["disease"].unique()

    # # 059GZ ch RSTN3-RSTN4 loc STN
    #compute_modality("059GZ", "fft")
    SHUFFLED = True
    
    PATH_BASE = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\0210\raw_new"
    PATH_BASE = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/EyesOpenBeijing/0210/raw_new'

    

    for idx_shuffle in tqdm(range(100)):
        if SHUFFLED:
            df_all["label_enc"] = np.random.permutation(df_all["label_enc"].values)
        for mod in modality_:  # 
            l_df_ = Parallel(n_jobs=len(subs))(delayed(compute_modality)(sub, mod) for sub in subs)
        
            df_per = pd.DataFrame(list(np.concatenate(l_df_)))
            if SHUFFLED:
                PATH_SAVE = os.path.join(PATH_BASE, f"out_per_loc_mod_{mod}_fft_with_lfa_CB_shuffled_{idx_shuffle}.csv")
            else:
                PATH_SAVE = os.path.join(PATH_BASE, f"out_per_loc_mod_{mod}_fft_with_lfa_CB.csv")
            df_per.to_csv(PATH_SAVE, index=False)



    # plt.figure()
    # sns.boxplot(data=df_per, x="loc", y="ba", hue="mod", palette="viridis")
    # #sns.swarmplot(data=df_per, x="loc", y="ba", color=".25", hue="mod")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.savefig("ba_")
    # plt.show(block=True)

    # plt.figure()
    # sns.boxplot(data=df_per.query("mod == 'all'"), x="disease", y="ba", hue="mod", palette="viridis")
    # sns.swarmplot(data=df_per.query("mod == 'all'"), x="disease", y="ba", color=".25", hue="mod")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.savefig("ba_")
    # plt.show(block=True)

    # for loc in locs:
    #     features = ["gamma",]
    #     #if loc == "STN":
    #     #    features = ["alpha", "LSTN1-LSTN2", "RSTN3-RSTN3"]
    #     features = features + [loc]

    #     df = df_all.copy()
        
    #     # integer encode label column
    #     df["label_enc"] = df["label"].map({"SLEEP": 0, "EyesOpen": 1, "EyesClosed": 2})
    #     df = df.drop(columns=["time", "label"])

    #     # run a three fold cross validation to decode the label column, remove the time column
    #     cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    #     model = linear_model.LinearRegression()
    #     cm_list = []
    #     ba_list = []
    #     for train_, test_ in cv.split(df, df["label_enc"]):
    #         # integer encode the label column
    #         train_df = df.iloc[train_]
    #         train_df_balanced = balance_classes(train_df, "label_enc")

    #         X_train = train_df_balanced.drop(columns=["label_enc"])
    #         y_train = train_df_balanced["label_enc"]

    #         # ensure that the same number of samples are taken for training from each class

            
    #         X_test = df.iloc[test_, :]
    #         y_test = df.iloc[test_, :]["label_enc"]
    #         X_test = X_test.drop(columns=["label_enc"])
            
    #         # include only columns that where all feature_list elements needs to be in the channel name
    #         cols_use = [c for c in X_train.columns if all([f in c for f in features])]
    #         X_train = X_train[cols_use]
    #         X_test = X_test[cols_use]
            

    #         model = linear_model.LogisticRegression(class_weight="balanced")
    #         model.fit(X_train, y_train)
    #         pred = model.predict(X_test)

            
    #         # plot the confusion matrix
    #         cm = metrics.confusion_matrix(y_test, pred, normalize="true")
    #         #disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SLEEP", "EyesOpen", "EyesClosed"])
    #         #disp.plot(cmap=plt.cm.Blues)
    #         #plt.title("Confusion Matrix")
    #         #plt.show(block=True)

    #         cm_list.append(cm)
    #         ba_list.append(metrics.balanced_accuracy_score(y_test, pred))

    #         print(metrics.balanced_accuracy_score(y_test, pred))
        
    #     # plot the mean confusion matrix
    #     cm_mean = sum(cm_list) / len(cm_list)
    #     #disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=["SLEEP", "EyesOpen", "EyesClosed"])
    #     #disp.plot(cmap=plt.cm.Blues)
    #     #plt.title("Mean Confusion Matrix")
    #     #plt.show(block=True)

    #     ba_mean = sum(ba_list) / len(ba_list)
    #     ca_loc.append(cm_mean)
    #     ba_loc.append(np.round(ba_mean, 2))

    # # make a subplot for each confusion matrix:
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed
    # for i, ax in enumerate(axs):
    #     disp = metrics.ConfusionMatrixDisplay(
    #         confusion_matrix=ca_loc[i],
    #         display_labels=["SLEEP", "EyesOpen", "EyesClosed"]
    #     )
    #     disp.plot(cmap=plt.cm.Blues, ax=ax)
    #     ax.set_title(f"loc: {locs[i]} Mean Confusion Matrix {ba_loc[i]}")
    # plt.tight_layout()
    # plt.show(block=True)