import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the csv as a dataframe
df = pd.read_csv(r"D:\Glenn\coeff_l1_2.csv")

df_perf = pd.read_csv(r"D:\Glenn\df_all_features.csv")

ch_full = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']

featuredim = ch_full['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[3] = np.add(idxlist_Berlin_001[3],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))

fields = ['cohort', 'sub', 'ch', 'type']

corrfeaturedim = [featuredim[i] for i in idxlist[-1]]
allfields = fields+ corrfeaturedim

cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]

idxofmax = np.sort(list(df_perf.groupby(['cohort','sub'])['ba_combined'].idxmax()))

## Heatmap of the importance per feature (relative color per subject (row))
plt.figure()
maxpersub = df.iloc[idxofmax][corrfeaturedim]
negatives = maxpersub<0 # Find negative values
absolutes = maxpersub.abs()
maxpersub_n = maxpersub.div(maxpersub.max(axis=1), axis=0)
maxpersub_n = maxpersub.subtract(maxpersub.mean(axis=1),axis=0).div(maxpersub.std(axis=1),axis=0)
maxpersub_01 = absolutes.subtract(absolutes.min(axis=1),axis=0).div(absolutes.max(axis=1).subtract(absolutes.min(axis=1)),axis=0)
maxpersub_01 = maxpersub_01* (negatives*-2+1)
subject = df.iloc[idxofmax]['sub']
ax = sns.heatmap(maxpersub_01,yticklabels= subject,annot=maxpersub_01,cmap="coolwarm")
ax.set(xlabel="", ylabel="")
ax.set_xticks(list(range(len(corrfeaturedim))))
ax.set_xticklabels(corrfeaturedim)
ax.xaxis.tick_top()
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

### Ranking of relative importance (take absolute value and then scale [0,1] per subject, then boxplot per feature
maxpersub = df.iloc[idxofmax][corrfeaturedim].abs()
maxpersub_n = maxpersub.subtract(maxpersub.min(axis=1),axis=0).div(maxpersub.max(axis=1)-maxpersub.min(axis=1),axis=0)
meds = maxpersub_n.median(axis=0)
meds = meds.sort_values(ascending=False, inplace=False)
ax = maxpersub_n[meds.index].plot(kind='box')
ax.set(title='Relative importance of features [0,1] over subjects')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

