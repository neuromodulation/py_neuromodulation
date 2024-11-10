import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import os
import seaborn as sns


def read_per(d_out):
    l = []
    for CLASSIFICATION in d_out.keys():
        if CLASSIFICATION is True:
            per_ = "ba"
        else:
            per_ = "corr_coeff"
        for pkg_label in d_out[CLASSIFICATION].keys():
            for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
                l.append({
                    "sub": sub,
                    "pkg_label": pkg_label,
                    "CLASSIFICATION": CLASSIFICATION,
                    "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
                })
    df_loso = pd.DataFrame(l)
    return df_loso

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
l_ = []
for exclude_hour in [True, False]:
    file = f"LOHO_ALL_LABELS_ALL_GROUPS_exludehour_{exclude_hour}.pkl"
    with open(os.path.join(PATH_PER, file), "rb") as f:
        d_out = pickle.load(f)
        df_ = read_per(d_out)
        df_["hour_feature"] = not exclude_hour
        l_.append(df_)
df_h = pd.concat(l_, axis=0)



l_ = []
for exclude_night in [True, False]:
    file = f"LOHO_ALL_LABELS_ALL_GROUPS_exludenight_{exclude_night}.pkl"
    with open(os.path.join(PATH_PER, file), "rb") as f:
        d_out = pickle.load(f)
        df_ = read_per(d_out)
        df_["include_night"] = not exclude_night
        l_.append(df_)
df_n = pd.concat(l_, axis=0)

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

plt.figure(figsize=(10, 7), dpi=300)
plt.subplot(2, 2, 1)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Balanced accuracy")
plt.subplot(2, 2, 2)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exlude_hour_feature.pdf"))
plt.subplot(2, 2, 3)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Balanced accuracy")
plt.subplot(2, 2, 4)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exclude_analysis.pdf"))
plt.show(block=True)
