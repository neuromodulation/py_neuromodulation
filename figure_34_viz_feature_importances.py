import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_PER = os.path.join(PATH_PER, "LOHO_ALL_LABELS_ALL_GROUPS_exludehour_False.pkl")

PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"

with open(PATH_PER, "rb") as f:
    d_out = pickle.load(f)

PATH_FEATURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
df_all = pd.read_csv(os.path.join(PATH_FEATURES, "all_merged_normed.csv"), index_col=0)
df_all = df_all.dropna(axis=1)
df_all = df_all.replace([np.inf, -np.inf], np.nan)
df_all = df_all.dropna(axis=1)
df_all = df_all.drop(columns=["sub",])
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
df_all["hour"] = df_all["pkg_dt"].dt.hour

# remove columns that start with pkg
df_all = df_all[[c for c in df_all.columns if not c.startswith("pkg")]]
columns_ = df_all.columns

plt.figure(figsize=(10, 10))
cols_show = 50
for idx_, pkg_decode_label in enumerate(["pkg_dk", "pkg_bk", "pkg_tremor"]):

    data = []
    if pkg_decode_label == "pkg_bk":
        CLASS_ = False
    else:
        CLASS_ = True
    d_out_ = d_out[CLASS_][pkg_decode_label]["ecog_stn"]
    for sub in d_out_.keys():
        data.append(d_out_[sub]["feature_importances"])
    fimp = np.array(data)
    mean_fimp = fimp.mean(axis=0)
    cols_sorted = np.array(columns_)[np.argsort(mean_fimp)[::-1]]

    plt.subplot(3, 1, idx_+1)
    plt.barh(cols_sorted[:cols_show], mean_fimp[np.argsort(mean_fimp)[::-1]][:cols_show])
    plt.gca().invert_yaxis()
    plt.title(pkg_decode_label)
    plt.xlabel("Feature importance - Prediction Value Change")
    #plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "feature_importance_plt_bar.pdf"))
plt.show(block=True)

