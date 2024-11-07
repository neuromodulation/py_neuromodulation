import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import seaborn as sns
import numpy as np

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"

def read_per_out(PATH_):
    with open(PATH_, "rb") as f:
        d_out = pickle.load(f)

    data = []

    if list(d_out.keys())[0].startswith("rcs"):
        if "pkg_bk" in PATH_:
            key_per = "corr_coeff"
            pkg_decode_label = "pkg_bk"
        elif "pkg_dk" in PATH_:
            pkg_decode_label = "pkg_dk"
            key_per = "ba"
        elif "pkg_tremor" in PATH_:
            pkg_decode_label = "pkg_tremor"
            key_per = "ba"
        for sub in d_out.keys():
            data.append({
                "sub": sub,
                "pkg_decode_label": pkg_decode_label,
                "per": d_out[sub][key_per],
            })
        df = pd.DataFrame(data)
        return df
    
    for pkg_decode_label in d_out.keys():
        for loc in d_out[pkg_decode_label].keys():
            for sub in d_out[pkg_decode_label][loc].keys():
                if pkg_decode_label == "pkg_bk":
                    data.append({
                        "sub" : sub,
                        "pkg_decode_label": pkg_decode_label,
                        "per": d_out[pkg_decode_label][loc][sub]["corr_coeff"],
                        #"r2" : d_out[pkg_decode_label][loc][sub]["r2"],
                        #"mae" : d_out[pkg_decode_label][loc][sub]["mae"],
                        #"mse" : d_out[pkg_decode_label][loc][sub]["mse"],
                    })
                else:
                    data.append({
                        "sub": sub,
                        "pkg_decode_label": pkg_decode_label,
                        #"f1": d_out[pkg_decode_label][loc][sub]["f1"],
                        "per": d_out[pkg_decode_label][loc][sub]["ba"],
                        
                    })

    df = pd.DataFrame(data)
    return df

def get_all_ch_performances(CLASSIFICATION, pkg_label, per_):
    PATH_PRE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/LOSO_ALL_LABELS_ALL_GROUPS.pkl'
    with open(PATH_PRE, "rb") as f:
        d_out = pickle.load(f)
    l = []
    #for CLASSIFICATION in d_out.keys():
    #    for pkg_label in d_out[CLASSIFICATION].keys():
    for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
        if CLASSIFICATION is True:
            per_ = "ba"
        else:
            per_ = "corr_coeff"
        l.append({
            "sub": sub,
            "pkg_decode_label": pkg_label,
            "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
        })
    df_loho = pd.DataFrame(l)
    return df_loho #df_loho["cv"] = "LOHO"

def get_dur_per_relation(label):
    if label == "pkg_dk":
        label_find = "_DK_"
    else:
        label_find = "_"+label+"_"
    files = [f for f in os.listdir(PATH_PER) if "LOHO" in f and"_min.pkl" in f and label_find in f]
    df_ = []
    for f in files:
        with open(os.path.join(PATH_PER, f), "rb") as f:
            d_out = pickle.load(f)

            l = []
            for CLASS in d_out.keys():
                for label_name in d_out[CLASS].keys():
                    for loc_ in d_out[CLASS][label_name].keys():
                        for sub_test in d_out[CLASS][label_name][loc_].keys():
                            if label_name == "pkg_dk" or label == "pkg_tremor":
                                per_label = "ba"
                            else:
                                per_label = "corr_coeff"
                            l.append({
                                "sub": sub_test,
                                "pkg_label": label_name,
                                "CLASS": CLASS,
                                "per": d_out[CLASS][label_name][loc_][sub_test][per_label]
                            })
            df = pd.DataFrame(l)
            df["dur"] = int(str(f).split("_")[-2])

            df_.append(df)

    df = pd.concat(df_, axis=0)
    # clip the balanced accuracy to 0.5 and 1
    if label != "pkg_bk":
        df["per"] = np.clip(df["per"], 0.5, 1)

    return df

def plot_boxplot(df, x_label, y_label="Balanced accuracy", order_ = None):
    sns.boxplot(x=x_label, y="per", data=df, showmeans=True, showfliers=False, palette="viridis", order=order_)
    #sns.swarmplot(x="norm_window", y="ba", data=df_all, color="black", alpha=0.5, palette="viridis")
    # put the mean values as text on top of the boxplot
    means = df.groupby(x_label)["per"].mean()
    if order_ is not None:
        for i, x_label_ in enumerate(df.groupby(x_label)["per"].mean().sort_values(ascending=True).index):
            mean = df[df[x_label] == x_label_]["per"].mean()
            plt.text(i, mean, f"{np.round(mean, 2)}", ha="center", va="bottom")
    else:
        for i, mean in enumerate(means):
            plt.text(i, mean, f"{mean:.2f}", ha="center", va="bottom")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.show(block=True)

def plot_per_train_time_relation(df, label):
        #plt.figure(figsize=(10, 5), dpi=300)
    durations = np.sort(df["dur"].unique())
    sub_per = []
    for sub in df["sub"].unique():
        df_sub = df.query(f"sub == '{sub}'")
        # sort the dataframe by duration
        df_sub = df_sub.sort_values("dur")
        plt.plot(durations / 60, df_sub["per"], color="gray", alpha=0.2)
        sub_per.append(df_sub["per"].values)
    plt.xlabel("Duration [h]")
    if label == "bk":
        plt.ylabel("Correlation coefficient")
    else:
        plt.ylabel("Balanced accuracy")
    # plot the mean accuracy for each duration
    plt.plot(durations / 60, np.array(sub_per).mean(axis=0), marker="o", linestyle="-", color="black")
    # write the mean accuracy on top of the line
    for i, dur in enumerate(durations):
        plt.text(durations[i] / 60, np.array(sub_per).mean(axis=0)[i], f"{np.round(np.array(sub_per).mean(axis=0)[i], 2)}", ha="center", va="bottom")
    
    plt.xscale('log')
    #plt.title(f"LOHO PKG BK CV different training duration")
    #plt.tight_layout()
    #plt.savefig(os.path.join(PATH_FIGURE, f"LOHO_different_training_duration_sub_{label}.pdf"))

if __name__ == "__main__":

    plt.figure(figsize=(10, 7))
    for idx_, label_name in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor"]):

        l_models = []
        for ML_ in ["CB", "LM", "XGB", "PCA_LM", "CEBRA", "RF"]:
            PATH_READ = os.path.join(PATH_PER, f"d_out_ML_across_patients_{label_name}_10s_seglength_480_all_{ML_}.pkl")
            df = read_per_out(PATH_READ)
            df["model"] = ML_
            l_models.append(df)
        df_models = pd.concat(l_models)

        mod_files = [f for f in os.listdir(PATH_PER) if f"d_out_patient_across_{label_name}_feature_mod" in f]
        mods = [f[f.find("feature_mod_")+len("feature_mod_"):f.find("_480")] for f in mod_files]

        l_features = []
        for mod_idx, mod in enumerate(mod_files):
            PATH_READ = os.path.join(PATH_PER, mod)
            df = read_per_out(PATH_READ)
            df["feature_mod"] = mods[mod_idx]
            l_features.append(df)
        
        df_features = pd.concat(l_features, axis=0)
        if label_name == "pkg_bk":
            df_all_features = get_all_ch_performances(False, label_name, "corr_coeff")
        else:
            df_all_features = get_all_ch_performances(True, label_name, "ba")
        df_all_features["feature_mod"] = "all"
        df_features_comb = pd.concat([df_features, df_all_features], axis=0)

        # now the second subplot: Normalization windows
        if label_name == "pkg_bk":
            class_ = "False"
        else:
            class_ = "True"
        l_norms = []
        for norm_window in [0, 5, 10, 20, 30, 60, 120, 180, 300, 480, 720, 960, 1200, 1440]:
            OUT_FILE = f"d_out_patient_across_{label_name}_class_{class_}_{norm_window}.pkl"
            PATH_READ = os.path.join(PATH_PER, OUT_FILE)
            if not os.path.exists(PATH_READ):
                continue

            df = read_per_out(PATH_READ)
            #df = df.query("loc == 'ecog_stn'")
            df["norm_window"] = norm_window
            l_norms.append(df)
        df_norm = pd.concat(l_norms)

        df_per_dur_rel = get_dur_per_relation(label_name)

        if label_name == "pkg_bk":
            y_label = "Correlation coefficient"
        else:
            y_label = "Balanced accuracy"
        plt.subplot(3, 4, 4*idx_+1)
        plot_boxplot(df_norm, "norm_window", y_label)
        
        plt.subplot(3, 4, 4*idx_+2)
        plot_boxplot(df_features_comb, "feature_mod", y_label,
                    order_=df_features_comb.groupby("feature_mod")["per"].mean().sort_values(ascending=True).index)
        
        plt.subplot(3, 4, 4*idx_+3)
        plot_boxplot(df_models, "model", y_label,
                    order_=df_models.groupby("model")["per"].mean().sort_values(ascending=True).index)
        
        plt.subplot(3, 4, 4*idx_+4)
        plot_per_train_time_relation(df_per_dur_rel, label_name)

    #plt.savefig(os.path.join(PATH_FIGURES, "figure_33_joint_plot.pdf"))
    plt.show(block=True)

    print("df")