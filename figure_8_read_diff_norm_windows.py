import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle

updrs_ = {
    "rcs02": 4,
    "rcs05": 1,
    "rcs06": 2,
    "rcs07": 1,
    "rcs08": 2,
    "rcs09": 2,
    "rcs10": 0,
    "rcs11": 2,
    "rcs12": 2,
    "rcs14": 0,
    "rcs15": 0,
    "rcs17": 1,
    "rcs18": 2,
    "rcs19": 0,
    "rcs20": 1,
    "rcs03": 1
}

def read_pkg_out(PATH_):
    with open(PATH_, "rb") as f:
        d_out = pickle.load(f)

    data = []
    for pkg_decode_label in d_out.keys():
        for loc in d_out[pkg_decode_label].keys():
            for sub in d_out[pkg_decode_label][loc].keys():
                data.append({
                    "accuracy": d_out[pkg_decode_label][loc][sub]["accuracy"],
                    "f1": d_out[pkg_decode_label][loc][sub]["f1"],
                    "ba": d_out[pkg_decode_label][loc][sub]["ba"],
                    "sub": sub,
                    "pkg_decode_label": pkg_decode_label,
                    "loc": loc,
                })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":

    PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
    PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"

    l_norms = []
    for norm_window in [5, 10, 20, 30, 60, 120, 180, 300, 480, 720, 960, 1200, 1440]:
        OUT_FILE = f"d_out_patient_across_class_{norm_window}.pkl"
        PATH_READ = os.path.join(PATH_PER, OUT_FILE)

        df = read_pkg_out(PATH_READ)
        df = df.query("loc == 'ecog_stn'")
        df["norm_window"] = norm_window
        l_norms.append(df)
    df_all = pd.concat(l_norms)

    plt.figure(figsize=(5, 5), dpi=300)
    sns.boxplot(x="norm_window", y="ba", data=df_all, showmeans=True, showfliers=False, palette="viridis")
    #sns.swarmplot(x="norm_window", y="ba", data=df_all, color="black", alpha=0.5, palette="viridis")
    # put the mean values as text on top of the boxplot
    means = df_all.groupby("norm_window")["ba"].mean()
    for i, mean in enumerate(means):
        plt.text(i, mean, f"{mean:.2f}", ha="center", va="bottom")

    plt.xlabel("Normalization window [min]")
    plt.ylabel("Balanced accuracy")
    plt.title("Different normalization windows")
    plt.tight_layout()
    plt.show(block=True)
