import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

leaveacross = True

suffix = '2023_10_16-11_33'

leave_cohort_out = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_cohort_out.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_within = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_sub_out_within_coh.npy"),
    allow_pickle="TRUE",
).item()
if leaveacross:
    leave_sub_across = np.load(
        os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_sub_out_across_coh.npy"),
        allow_pickle="TRUE",
    ).item()
    listofdicts = [leave_cohort_out, leave_sub_across, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_across_cohorts", "leave_subject_within_cohort"]
else:
    listofdicts = [leave_cohort_out, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_within_cohort"]


Col = ['Cohort','Validation','Performance']
# Repeat this over
df = pd.DataFrame(columns=['Cohort','Validation','Performance'])
for cohort in ['Beijing','Pittsburgh','Berlin','Washington']:
    for valset in range(len(valtypes)):
        dataset = listofdicts[valset]
        result = [[cohort,valtypes[valset],dataset[cohort][item]['performance']] for item in dataset[cohort] if isinstance(dataset[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result,columns=Col)
        df = pd.concat([df,df_temp])

suffix_log = '2023_10_16-11_29'

leave_cohort_out = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_log}_leave_1_cohort_out.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_within = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_log}_leave_1_sub_out_within_coh.npy"),
    allow_pickle="TRUE",
).item()
if leaveacross:
    leave_sub_across = np.load(
        os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_log}_leave_1_sub_out_across_coh.npy"),
        allow_pickle="TRUE",
    ).item()
    listofdicts = [leave_cohort_out, leave_sub_across, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_across_cohorts", "leave_subject_within_cohort"]
else:
    listofdicts = [leave_cohort_out, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_within_cohort"]


Col = ['Cohort','Validation','Performance']
# Repeat this over
df_old = pd.DataFrame(columns=['Cohort','Validation','Performance'])
for cohort in ['Beijing','Pittsburgh','Berlin','Washington']:
    for valset in range(len(valtypes)):
        dataset = listofdicts[valset]
        result = [[cohort,valtypes[valset],dataset[cohort][item]['performance']] for item in dataset[cohort] if isinstance(dataset[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result,columns=Col)
        df_old = pd.concat([df_old,df_temp])


#sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
sns.catplot(data=df, x="Validation", y="Performance", kind="box",color="0.9")
sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
              size=4, edgecolor="black",dodge=True).set_title(f"{suffix}")
plt.ylim(0.3, 1)
plt.title('Logreg on new fft data')
plt.show()

sns.catplot(data=df_old, x="Validation", y="Performance", kind="box",color="0.9")
sns.swarmplot(x="Validation", y="Performance", data=df_old, hue="Cohort",
              size=4, edgecolor="black",dodge=True).set_title(f"{suffix_log}")
plt.ylim(0.3, 1)
plt.title('Logreg on old fft data')
plt.show()

df['Performance_log'] = df_old['Performance']

df_coh = df.loc[df['Validation'] == valtypes[0]]
data = pd.melt(df_coh[['Performance_log','Performance']])
data['nr'] = list(range(39))+list(range(39))

plt.figure()
sns.boxplot(x="variable", y="value", data=data)
sns.swarmplot(x="variable", y="value", data=data, color=".25")
pal = sns.color_palette(['black'], df.shape[1])
sns.lineplot(x="variable", y="value", hue='nr', data=data,
             estimator=None, legend=False,palette=pal)

df['diff'] = df['Performance'] - df['Performance_log']

## Analyze the mean difference per category
plt.figure()
df.groupby(['Cohort','Validation']).mean()['diff'].plot(kind='bar')
plt.title('Difference between old and new dataset with only fft and logistic regression')
plt.ylabel('New-Old fft Logistic Regression BA')
plt.figure()
sns.boxplot(x="Cohort", y="diff", data=df,hue='Validation',showmeans=True,meanline=True,meanprops={'color': 'r'})
plt.title('Difference between old and new dataset with only fft and logistic regression')
plt.ylabel('New-Old fft Logistic Regression BA')