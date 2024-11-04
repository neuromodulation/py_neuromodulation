import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
PATH_FIGURE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'

for label in ["bk", "tremor", "DK",]:
    files = [f for f in os.listdir(PATH_PER) if "LOHO" in f and"_min.pkl" in f and "_"+label+"_" in f]
    df_ = []
    for f in files:
        with open(os.path.join(PATH_PER, f), "rb") as f:
            d_out = pickle.load(f)

            l = []
            for CLASS in d_out.keys():
                for label_name in d_out[CLASS].keys():
                    for loc_ in d_out[CLASS][label_name].keys():
                        for sub_test in d_out[CLASS][label_name][loc_].keys():
                            if label == "DK" or label == "tremor":
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
    if label != "bk":
        df["per"] = np.clip(df["per"], 0.5, 1)

    df.groupby("dur")["per"].mean()

    # make a line plot for each subject and each duration
    plt.figure(figsize=(10, 5), dpi=300)
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
    plt.title(f"LOHO PKG {label} CV different training duration")
    plt.tight_layout()
    #plt.savefig(os.path.join(PATH_FIGURE, f"LOHO_different_training_duration_sub_{label}.pdf"))
    plt.show(block=True)
    print("")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="dur", y="per", palette="viridis", showfliers=False, showmeans=True)
sns.swarmplot(data=df, x="dur", y="per", palette="viridis", alpha=0.5, dodge=False, size=2.5)
plt.xlabel("Duration [min]")
plt.ylabel("Balanced accuracy")
plt.title("LOHO PKG DK CLASS CV different training duration")
# write the mean accuracy on top of the boxplot
for i, dur in enumerate(np.sort(df["dur"].unique())):
    plt.text(i, df.query(f"dur == {dur}")["per"].mean(), f"{np.round(df.query(f'dur == {dur}')['per'].mean(), 2)}", ha="center", va="bottom")
plt.savefig(os.path.join(PATH_FIGURE, "LOHO_different_training_duration.pdf"), dpi=300)
plt.show(block=True)

