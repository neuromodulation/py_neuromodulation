import numpy as np
import os
import pandas as pd

import seaborn as sns

leave_cohort_out = np.load(
    os.path.join(r"D:\Glenn\CEBRA performances", "out_per_offset10_leave_1_cohort_out.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_across = np.load(
    os.path.join(r"D:\Glenn\CEBRA performances", "out_per_offset10_leave_1_sub_out_across_coh.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_within = np.load(
    os.path.join(r"D:\Glenn\CEBRA performances", "out_per_offset10_leave_1_sub_out_within_coh.npy"),
    allow_pickle="TRUE",
).item()
listofdicts = [leave_cohort_out, leave_sub_across, leave_sub_within]
valtypes = ["leave_cohort_out", "leave_subject_across_cohorts", "leave_subject_within_cohort"]
# Repeat this over
df = pd.DataFrame(columns=['Cohort','Validation','Performance'])
for cohort in ['Beijing','Pittsburgh','Berlin']:
    for valset in range(3):
        dataset = listofdicts[valset]
        result = [[cohort,valtypes[valset],dataset[cohort][item]['performance']] for item in dataset[cohort]]
        df_temp = pd.DataFrame(data=result,columns=['Cohort','Validation','Performance'])
        df = pd.concat([df,df_temp])


#sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
sns.catplot(data=df, x="Validation", y="Performance", kind="box",color="0.9")
sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
              size=4, edgecolor="black",dodge=True)