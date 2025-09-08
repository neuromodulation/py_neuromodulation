import pandas as pd
import os
from scipy.stats import friedmanchisquare
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from py_neuromodulation import nm_stats
from scipy import stats

PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\0210\raw_new"
PATH_ = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/EyesOpenBeijing/0210/raw_new'

PATH_FIGURES = os.path.join(PATH_, "figures")
fontsize_ = 10
mods = ["alpha", "fft"][::-1]

# 059GZ ch RSTN3-RSTN4 loc STN

l_ = []
for mod in mods:
    df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_{mod}_fft_with_lfa.csv"))
    df["mod"] = mod
    l_.append(df)

df_all = pd.concat(l_)

df_all["mod"] = df_all["mod"].replace("low_frequency_activity", "alpha")
df_all["dout"] = df_all["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
df_all = df_all.query("dout != 'HD'").query("dout != 'TS'")

# 1. Plot: Performance low frequency activity vs fft
plt.figure(figsize=(2,4), dpi=300)
sns.boxplot(data=df_all.groupby(["sub", "mod"])["ba"].max().reset_index(),
            x="mod", y="ba", color="#46C389", showfliers=False, showmeans=False, width=0.6,
            order=["alpha", "fft"])
sns.swarmplot(data=df_all.groupby(["sub", "mod"])["ba"].max().reset_index(),
              x="mod", y="ba", color=".25", alpha=0.5, dodge=True, size=4, order=["alpha", "fft"])
plt.xlabel("F-band", fontsize=fontsize_)
plt.ylabel("Balanced accuracy", fontsize=fontsize_)
plt.xticks(fontsize=fontsize_); plt.yticks(fontsize=fontsize_)
plt.ylim(0.5, 1)
plt.title("Best channels", fontsize=fontsize_)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_lfa_vs_all.pdf"))
plt.show(block=True)

# STATS:
def print_mean_std(s: pd.Series):
    print(f"Mean: {s.mean():.2f} +/- {s.std():.2f}")

group_lfa = df_all.groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
group_all = df_all.groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
print_mean_std(group_lfa)  # 0.67 +/- 0.09
print_mean_std(group_all) # 0.82 +/- 0.11
print(stats.mannwhitneyu(group_lfa, group_all)) # p<10^-5

# 2. Plot: Performance low frequency activity vs fft per disease
plt.figure(figsize=(3,4), dpi=300)
diseases_ = df_all.dout.unique()
# select only PD and Dys diseases
sns.boxplot(data=df_all.groupby(["sub", "mod", "dout"])["ba"].max().reset_index(),
            x="mod", y="ba", hue="dout", palette="viridis", showfliers=False, showmeans=False,
            order=["alpha", "fft"], width=0.6)
ax = sns.swarmplot(data=df_all.groupby(["sub", "mod", "dout"])["ba"].max().reset_index(),
                   x="mod", y="ba", hue="dout", color=".25", alpha=0.5, dodge=True, size=3, 
                   order=["alpha", "fft"])
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(fontsize=fontsize_); plt.yticks(fontsize=fontsize_)
plt.ylabel("Balanced accuracy", fontsize=fontsize_)
plt.xlabel("F-band", fontsize=fontsize_)
plt.ylim(0.5, 1)
plt.title("Diseases", fontsize=fontsize_)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_diseases.pdf"))
plt.show(block=True)

group_PD = df_all.query("dout == 'PD'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
group_Dys = df_all.query("dout == 'Dys'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
print_mean_std(group_PD)  # 0.82 +/- 0.11
print_mean_std(group_Dys) # 0.82 +/- 0.11
print(stats.mannwhitneyu(group_PD, group_Dys)) # p=0.86

group_PD = df_all.query("dout == 'PD'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
group_Dys = df_all.query("dout == 'Dys'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
print_mean_std(group_PD)  # 0.69 +/- 0.11
print_mean_std(group_Dys) # 0.65 +/- 0.08
print(stats.mannwhitneyu(group_PD, group_Dys))  # p=0.27

# 3. Plot: Performance low frequency activity vs fft per location
plt.figure(figsize=(3, 4), dpi=300)
sns.boxplot(data=df_all.groupby(["sub", "mod", "loc"])["ba"].max().reset_index(),
            x="mod", y="ba", hue="loc", palette="viridis", showfliers=False, showmeans=False,
            order=["alpha", "fft"], width=0.6)
ax = sns.swarmplot(data=df_all.groupby(["sub", "mod", "loc"])["ba"].max().reset_index(),
                   x="mod", y="ba", hue="loc", color=".25", alpha=0.5, dodge=True, size=3, 
                   order=["alpha", "fft"])
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel("Balanced accuracy", fontsize=fontsize_)
plt.xlabel("F-band", fontsize=fontsize_)
plt.xticks(fontsize=fontsize_); plt.yticks(fontsize=fontsize_)
plt.ylim(0.5, 1)
plt.title("Locations", fontsize=fontsize_)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_locations.pdf"))
plt.show(block=True)

group_STN = df_all.query("loc == 'STN'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
group_GPI = df_all.query("loc == 'GPI'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
print_mean_std(group_STN)  # 0.88 +/- 0.08
print_mean_std(group_GPI) # 0.77 +/- 0.11
print(stats.mannwhitneyu(group_STN, group_GPI))  # p=0.001

group_STN = df_all.query("loc == 'STN'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
group_GPI = df_all.query("loc == 'GPI'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
print_mean_std(group_STN)  # 0.71 +/- 0.1
print_mean_std(group_GPI) # 0.62 +/- 0.06
print(stats.mannwhitneyu(group_STN, group_GPI))  # p=0.001

# 4. Plot: Coefficients

def group_coef_df(group_mod):
    df_coef = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_fft_fft_with_lfa.csv"))
    # replace coef_low column with coef_lfa
    df_coef["coef_lfa"] = df_coef["coef_low"]
    df_coef = df_coef.drop(columns=["coef_low"])
    # tile the dataframe to have all coef_ columns in one column
    df_coef_melt = df_coef.melt(id_vars=["sub", "ch","mod", group_mod, "ba"], value_vars=cols_coef)
    df_coef_melt_best = df_coef_melt.groupby(["sub", "mod", "variable", group_mod])[["ba", "value"]].max("ba").reset_index()  #"loc", 
    if group_mod == "dout":
        df_coef_melt_best = df_coef_melt_best.query("dout != 'HD'").query("dout != 'TS'")
        df_coef_melt_best["dout"] = df_coef_melt_best["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
    return df_coef_melt_best

# loc location
plt.figure(figsize=(6, 4), dpi=300)
plt_list = ["loc", "dout"]
cols_coef = ['coef_theta', 'coef_alpha', 'coef_lfa', 'coef_low beta', 'coef_high beta', 'coef_low gamma', 'coef_high gamma', 'coef_HFA', ]

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.axhline(0, color="gray", alpha=0.4)
    df_coef_melt_best = group_coef_df(plt_list[i])
    sns.boxplot(data=df_coef_melt_best, x="variable", y="value", hue=plt_list[i], palette="viridis",
                order=cols_coef, showfliers=False, showmeans=False, width=0.6)
    ax = sns.swarmplot(data=df_coef_melt_best, x="variable", y="value", hue=plt_list[i], color=".25",
                    order=cols_coef, alpha=0.5, dodge=True, size=3)
    if i==0:
        plt.ylabel("Coefficients", fontsize=fontsize_)
    else:
        plt.ylabel("")
    plt.xticks(range(len(cols_coef)), [f[5:] for f in cols_coef], rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], )  # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.
    if i==0:
        plt.title("Locations", fontsize=fontsize_)
    else:
        plt.title("Diseases", fontsize=fontsize_)
    plt.xlabel("F-band", fontsize=fontsize_)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_coefficients.pdf"))
plt.show(block=True)


# 5. Plot: Performance Three class classification
df_all = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_fft_fft_with_lfa_three_classes.csv"))
df_all["mod"] = df_all["mod"].replace("low_frequency_activity", "lfa")
df_all["dout"] = df_all["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
df_all_diseases = df_all.query("dout != 'HD'").query("dout != 'TS'")

plt.figure(figsize=(2, 4), dpi=300)
ax = sns.boxplot(data=df_all_diseases.groupby(["sub", "mod", "dout"])["ba"].max().reset_index(),
            x="dout", y="ba", showfliers=False, showmeans=False,
            width=0.6, color="#46C389"
           )
ax = sns.swarmplot(data=df_all_diseases.groupby(["sub", "mod", "dout"])["ba"].max().reset_index(),
                   x="dout", y="ba", color=".25", alpha=0.5, dodge=True, size=3, 
                   )
plt.title("Three-classes", fontsize=fontsize_)
plt.ylabel("Balanced accuracy", fontsize=fontsize_)
plt.axhline(0.33, color="gray", alpha=0.4)
plt.ylim(0.2, 1)
plt.xlabel("Disease", fontsize=fontsize_)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_three_classes.pdf"))
plt.show(block=True)

group_GPI = df_all_diseases.query("loc == 'GPI'").groupby("sub")["ba"].max()
group_STN = df_all_diseases.query("loc == 'STN'").groupby("sub")["ba"].max()
print_mean_std(group_GPI)  # 0.7 +/- 0.12
print_mean_std(group_STN) # 0.81 +/- 0.07
print(stats.mannwhitneyu(group_STN, group_GPI))  # p=0.0046

group_PD = df_all_diseases.query("dout == 'PD'").groupby("sub")["ba"].max()
group_Dys = df_all_diseases.query("dout == 'Dys'").groupby("sub")["ba"].max()
print_mean_std(group_PD)  # 0.75 +/- 0.1
print_mean_std(group_Dys) # 0.76 +/- 0.13
print(stats.mannwhitneyu(group_PD, group_Dys))  # p=0.57

# 6. Plot ML model comparison
df_CB_all_fbands = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_fft_fft_with_lfa_CB.csv"))
df_CB_all_fbands["mod"] = "fft"
df_CB_lfa = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_low_frequency_activity_fft_with_lfa_CB.csv"))
df_CB_lfa["mod"] = "alpha"

df_CB_all = pd.concat([df_CB_all_fbands, df_CB_lfa])
df_CB_all["model"] = "CB"

df_LM_all = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_low_frequency_activity_fft_with_lfa.csv"))
df_LM_all["mod"] = "alpha"
df_LM_all_fbands = pd.read_csv(os.path.join(PATH_, "out_per_loc_mod_fft_fft_with_lfa.csv"))
df_LM_all_fbands["mod"] = "fft"

df_LM_all = pd.concat([df_LM_all, df_LM_all_fbands])
df_LM_all["model"] = "LM"
df_all = pd.concat([df_CB_all, df_LM_all])

df_all["dout"] = df_all["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
df_all_diseases = df_all.query("dout != 'HD'").query("dout != 'TS'")


plt.figure(figsize=(2, 4), dpi=300)
ax = sns.boxplot(data=df_all_diseases.groupby(["sub", "mod", "dout", "model"])["ba"].max().reset_index(),
            x="mod", y="ba", hue="model", palette="viridis", showfliers=False, showmeans=False,
            width=0.6, order=["alpha", "fft"], hue_order=["LM", "CB"]
           )
ax = sns.swarmplot(data=df_all_diseases.groupby(["sub", "mod", "dout", "model"])["ba"].max().reset_index(),
                   x="mod", y="ba", hue="model", color=".25", alpha=0.5, dodge=True, size=3, 
                   order=["alpha", "fft"], hue_order=["LM", "CB"]
                   )
plt.title("Models", fontsize=fontsize_)
plt.ylabel("Balanced accuracy", fontsize=fontsize_)
plt.xlabel("F-band", fontsize=fontsize_)
plt.ylim(0.5, 1)
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], )  # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "ba_models.pdf"))
plt.show(block=True)

group_LM = df_all_diseases.query("model == 'LM'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
group_CB = df_all_diseases.query("model == 'CB'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'fft'")["ba"]
print_mean_std(group_LM)  # 0.82 +/- 0.11
print_mean_std(group_CB) # 0.86 +/- 0.11
#nm_stats.permutationTest(group_LM, group_CB, p=5000, plot_distr=False, )
nm_stats.permutationTest_relative(np.array(group_LM), np.array(group_CB), False, None, 5000)
print(stats.mannwhitneyu(group_LM, group_CB))  # p=0.16

# a relative permuation test shows that there is a significant difference between the two models
# p<10^-5, mean diff = 0.03, std diff = 0.03

group_LM = df_all_diseases.query("model == 'LM'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
group_CB = df_all_diseases.query("model == 'CB'").groupby(["sub", "mod"])["ba"].max().reset_index().query("mod == 'alpha'")["ba"]
print_mean_std(group_LM)  # 0.82 +/- 0.11
print_mean_std(group_CB) # 0.86 +/- 0.11
#nm_stats.permutationTest(group_LM, group_CB, p=5000, plot_distr=False, )
nm_stats.permutationTest_relative(np.array(group_LM), np.array(group_CB), False, None, 5000)
print(stats.mannwhitneyu(group_LM, group_CB))  # p=0.16

# PREVIOUS:

for disease_idx, disease in enumerate(diseases_):
    for loc_idx, loc_ in enumerate(locs_):
        df_disease = df_all[df_all["dout"] == disease]
        df_disease = df_disease[df_disease["loc"] == loc_]
        plt.subplot(len(diseases_), len(locs_) , len(locs_)*disease_idx + 1 + loc_idx)
        sns.boxplot(data=df_disease, x="mod", y="ba", palette="viridis")
        plt.title(f"{loc_} {disease}")
        #df_disease = df_all[df_all["dout"] == disease]
        #plt.subplot(1, len(diseases_), disease_idx + 1)
        #sns.boxplot(data=df_all, x="loc", y="ba", hue="mod", palette="viridis")
plt.tight_layout()
plt.show(block=True)

df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_fft.csv"))
# melt the dataframe that all columns with coef_ become a column
df_melt = df.melt(id_vars=["ba", "loc", "sub", "dout"], value_vars=[c for c in df.columns if "coef_" in c])

plt.figure(figsize=(15, 12))
# melt the dataframe that coef_ becomes a column
sns.boxplot(data=df_melt, x="variable", y="value", palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("Coef")
plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(15, 12))
for disease_idx, disease in enumerate(diseases_):
    for loc_idx, loc_ in enumerate(locs_):
        df_disease = df_melt[df_melt["dout"] == disease]
        df_disease = df_disease[df_disease["loc"] == loc_]
        plt.subplot(len(diseases_), len(locs_) , len(locs_)*disease_idx + 1 + loc_idx)
        sns.boxplot(data=df_disease, x="variable", y="value", palette="viridis")
        plt.title(f"{loc_} {disease}")
        #df_disease = df_all[df_all["dout"] == disease]
        #plt.subplot(1, len(diseases_), disease_idx + 1)
        #sns.boxplot(data=df_all, x="loc", y="ba", hue="mod", palette="viridis")
plt.tight_layout()
plt.show(block=True)


#  .groupby(["sub"]).max()

plt.figure(figsize=(4, 3), dpi=300)
sns.boxplot(data=df.query("loc == 'STN'"), x="dout", y="ba")
sns.swarmplot(data=df.query("loc == 'STN'"), x="dout", y="ba", color="gray", alpha=0.5)
plt.title("STN sleep-eyes open-eye closed alpha only")
plt.tight_layout()
plt.show(block=True)