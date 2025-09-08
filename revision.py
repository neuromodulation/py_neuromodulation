import pandas as pd

PATH_FEATURES = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/EyesOpenBeijing/0210/raw_new/features_all_with_lfa.csv'


# create figure of shuffled performances
import pandas as pd
import os
from scipy.stats import friedmanchisquare
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from py_neuromodulation import nm_stats
from scipy import stats

PATH_ = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/EyesOpenBeijing/0210/raw_new'

PATH_FIGURES = os.path.join(PATH_, "figures")
fontsize_ = 10
mods = ["alpha", "fft"][::-1]

# 059GZ ch RSTN3-RSTN4 loc STN

l_ = []
for mod in mods:
    df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_{mod}_fft_with_lfa_CB_shuffled_0.csv"))
    df["mod"] = mod
    l_.append(df)

df_all = pd.concat(l_)

df_all["mod"] = df_all["mod"].replace("low_frequency_activity", "alpha")
df_all["dout"] = df_all["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
df_all = df_all.query("dout != 'HD'").query("dout != 'TS'")
df_shuffled = df_all.copy()
df_shuffled["shuffled"] = True

l_ = []
for mod in mods:
    df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_{mod}_fft_with_lfa_CB.csv"))
    df["mod"] = mod
    l_.append(df)

df_all = pd.concat(l_)

df_all["mod"] = df_all["mod"].replace("low_frequency_activity", "alpha")
df_all["dout"] = df_all["dout"].replace({"Meige": "Dys", "CD": "Dys", "GD": "Dys"})
df_all = df_all.query("dout != 'HD'").query("dout != 'TS'")
df_non_shuffled = df_all.copy()
df_non_shuffled["shuffled"] = False

idx_best_non_shuffled = df_non_shuffled.groupby(["sub", "mod"])["ba"].idxmax()
df_non_shuffled = df_non_shuffled.loc[idx_best_non_shuffled]

rows_non_shuffled = []
for row in df_non_shuffled.itertuples():
    sub = row.sub
    mod = row.mod
    ch = row.ch
    dout = row.dout
    row_matched = df_shuffled.query("sub == @sub & mod == @mod & ch == @ch & dout == @dout")
    rows_non_shuffled.append(row_matched)
df_shuffled = pd.concat(rows_non_shuffled)
df_all = pd.concat([df_non_shuffled, df_shuffled])

row_ = df_non_shuffled.query("mod == 'fft'")["ba"] - df_shuffled.query("mod == 'fft'")["ba"]
# check if all row_ are > 0
(row_ > 0)


# plot a histogram 
plt.figure()
sns.histplot(data=df_all.query("mod == 'fft'"), x="ba", hue="shuffled", element="step", stat="density", common_norm=False, bins=20)
# plot vertical lines at mean
mean_shuffled = df_all.query("mod == 'fft' & shuffled == True")["ba"].mean()
std_shuffled = df_all.query("mod == 'fft' & shuffled == True")["ba"].std()
mean_non_shuffled = df_all.query("mod == 'fft' & shuffled == False")["ba"].mean()
std_non_shuffled = df_all.query("mod == 'fft' & shuffled == False")["ba"].std()
plt.axvline(mean_shuffled, color="orange", linestyle="--", label=f"Mean shuffled: {mean_shuffled:.2f}")
plt.axvline(mean_non_shuffled, color="blue", linestyle="--", label=f"Mean non-shuffled: {mean_non_shuffled:.2f}")
plt.xlabel("Balanced Accuracy")
plt.ylabel("Density")
plt.title(f"Mean ba shuffled vs non-shuffled (fft)\n mean non-shuffled: {mean_non_shuffled:.2f} ± {std_non_shuffled:.2f}\n mean shuffled: {mean_shuffled:.2f} ± {std_shuffled:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "histogram_ba_shuffled_vs_non_shuffled_fft.pdf"))




df_all = pd.read_csv(PATH_FEATURES)
# num_subjects
num_subjects = df_all["sub"].nunique()
times_s_three_class = df_all.groupby("sub")["time"].count()
times_s_two_class = df_all.query("label != 'SLEEP'").groupby("sub")["time"].count()
times_5_fold_cv_num_training = (0.8 * times_s_two_class).round(1)
times_5_fold_cv_num_testing = (0.2 * times_s_two_class).round(1)

times_s_eyes_open = df_all.query("label == 'EyesOpen'").groupby("sub")["time"].count()
times_s_eyes_closed = df_all.query("label == 'EyesClosed'").groupby("sub")["time"].count()

# combine the series into a dataframe, the key is also sub
df_summary = pd.DataFrame({
    "Total Time [s]": times_s_three_class,
    "Time without sleep [s]": times_s_two_class,
    "Time eyes open [s]": times_s_eyes_open,
    "Time eyes closed [s]": times_s_eyes_closed,
    "Time 5-fold CV training [s]": times_5_fold_cv_num_training,
    "Time 5-fold CV testing [s]": times_5_fold_cv_num_testing,
}).reset_index()
df_summary.to_csv("summary_time_per_subject.csv", index=False)

