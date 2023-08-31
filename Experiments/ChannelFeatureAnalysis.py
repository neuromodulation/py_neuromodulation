import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the csv as a dataframe
df = pd.read_csv(r"D:\Glenn\df_all_features.csv")

perflist = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined','95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined', 'length', 'movsamples']
baonly = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined']
CIonly = ['95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined']
cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]
# Best performing channel and corresponding feature per subject
meanperfpersub = df.groupby('sub')[perflist].mean()
meanperfpersubpertype = df.groupby(['type','sub'])[perflist].mean()
meanperfpercohort = df.groupby('cohort')[perflist].mean()
# mean of cohort means
meanofmeans = meanperfpercohort.mean()

# Plot and save tables
##### Boxplots
# boxplot over all channels
ax = sns.boxplot(data = df.melt(id_vars='cohort',value_vars=baonly, var_name=''
                                       , value_name='balanced accuracy'), x = '', y = 'balanced accuracy', hue='cohort')
ax.set(title='Performance of features (all channels)')
ax.axhline(0.5,ls='--')
# Boxplot over best channels per subject
idxofmax = np.sort(list(df.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersub = df.iloc[idxofmax][['cohort','ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined']]
ax = sns.boxplot(data = maxpersub.melt(id_vars='cohort',value_vars=baonly, var_name=''
                                       , value_name='balanced accuracy'), x = '', y = 'balanced accuracy', hue='cohort')
ax.set(title='Performance of features (best channel per subject)')
ax.axhline(0.5, ls='--')

# ba_combined ECOG vs LFP over best channel per subject (per cohort)
f, axes = plt.subplots(1, 2)
idxofmax_ECOG = np.sort(list(df.groupby(['type','sub','cohort'])['ba_combined'].idxmax().T['ECOG']))
idxofmax_LFP = np.sort(list(df.groupby(['type','sub','cohort'])['ba_combined'].idxmax().T['LFP']))
maxpersub_ECOG = df.iloc[idxofmax_ECOG][['cohort','type', 'ba_combined']]
maxpersub_LFP = df.iloc[idxofmax_LFP][['cohort','type', 'ba_combined']]

sns.boxplot(data = maxpersub_ECOG.melt(id_vars='cohort',value_vars='ba_combined', var_name='ECOG', value_name='balanced accuracy'), x = 'ECOG',
            y = 'balanced accuracy', hue='cohort', ax=axes[0]).set(title='Best ECOG channels')
sns.boxplot(data = maxpersub_LFP.melt(id_vars='cohort',value_vars='ba_combined', var_name='LFP', value_name='balanced accuracy'), x = 'LFP',
            y = 'balanced accuracy', hue='cohort', ax=axes[1]).set(title='Best LFP channels')
axes[0].set(ylim=[0.5, 1])
axes[1].set(ylim=[0.5, 1])

# Pie chart of ECOG vs LFP best performers
total = len(idxofmax)
def formatter(x):
    return f"{total*x/100:.0f} ({x:.2f})%"
typeofmax = df.iloc[idxofmax]['type'].value_counts().plot(kind='pie',autopct=formatter)
typeofmax.set(title='Type of the best performing channel')

##### Stats over the best performing channels (per sub)
maxpersub = df.iloc[idxofmax][baonly]
subject = df.iloc[idxofmax]['sub']
ax = sns.heatmap(maxpersub,yticklabels= subject,annot=maxpersub,cmap="viridis")
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

# Sort also on fft to check if that changes some subjects
idxofmaxfft = np.sort(list(df.groupby(['sub', 'cohort'])['ba_fft'].idxmax()))
maxpersub = df.iloc[idxofmaxfft][baonly]
subject = df.iloc[idxofmaxfft]['sub']
ax = sns.heatmap(maxpersub,yticklabels= subject,annot=maxpersub,cmap="viridis")
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

#### Mean performance per channel type (over all)
meanperfpertype = df.groupby('type')[baonly].mean().plot(kind= 'bar')

# Mean performance per cohort (over all)
meanperfpercohort[baonly].plot(kind='bar')

# Mean performance per subject (over all channel types)
scaled_df = (meanperfpersub - meanperfpersub.min(axis=0))/(meanperfpersub.max(axis=0) - meanperfpersub.min(axis=0))
ax = sns.heatmap(scaled_df,annot=meanperfpersub,cmap="viridis")
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

# Mean performance per subject (split in ECOG and LFP)
f, axes = plt.subplots(1, 2)
meanperfpersubECOG = meanperfpersubpertype.T['ECOG'].transpose()
scaled_df = (meanperfpersubECOG - meanperfpersubECOG.min(axis=0))/(meanperfpersubECOG.max(axis=0) - meanperfpersubECOG.min(axis=0))
sns.heatmap(scaled_df,annot=meanperfpersubECOG,cmap="viridis",ax=axes[0])
axes[0].set(xlabel="", ylabel="")
axes[0].xaxis.tick_top()
axes[0].set_xticks(axes[0].get_xticks())
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='left')

meanperfpersubLFP = meanperfpersubpertype.T['LFP'].transpose()
scaled_df = (meanperfpersubLFP - meanperfpersubLFP.min(axis=0))/(meanperfpersubLFP.max(axis=0) - meanperfpersubLFP.min(axis=0))
sns.heatmap(scaled_df,annot=meanperfpersubLFP,cmap="viridis",ax=axes[1])
axes[1].set(xlabel="", ylabel="")
axes[1].xaxis.tick_top()
axes[1].set_xticks(axes[1].get_xticks())
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='left')

# All stats per cohort
scaled_df = (meanperfpercohort - meanperfpercohort.min(axis=0))/(meanperfpercohort.max(axis=0) - meanperfpercohort.min(axis=0))
ax = sns.heatmap(scaled_df,annot=meanperfpercohort,cmap="viridis")
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

# Plot mean of means
plt.errorbar(x = baonly,y = meanofmeans[baonly], yerr=list(meanofmeans[CIonly]))