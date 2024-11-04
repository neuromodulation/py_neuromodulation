import pandas as pd    

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"

PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
df = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)
locs_new = []
for idx, row in df.iterrows():
    if row["loc"] == "ECOG":
        if row["ch_orig"] == "8-9" or row["ch_orig"] == "8-10":
            locs_new.append("SC")
        else:
            locs_new.append("MC")
    else:
        locs_new.append(row["loc"])
df["loc"] = locs_new
hue_order = ["STN", "GP", "SC", "MC"]

plt.figure(figsize=(5, 5), dpi=300)
sns.boxplot(x="label", y="per", hue="loc", hue_order=hue_order,
            data=df.query("classification == True"), palette="viridis", showfliers=False,
            showmeans=True)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=hue_order, data=df.query("classification == True"),
              palette="viridis", alpha=0.5, dodge=True, size=2.5)
plt.ylabel("Balanced accuracy")
plt.title("Region-wise performances")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "region_wise_performances.pdf"))
plt.show(block=True)

