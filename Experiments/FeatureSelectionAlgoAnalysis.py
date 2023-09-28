import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

# Load the csv as a dataframe
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\RecursivefeatureeliminationPerformance.csv")
perflist = ['cohort', 'sub', 'ch', 'type', 'ba_combined', 'importance', '95%CI_combined', 'length', 'movsamples']
baonly = ['ba_combined']
cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]

# Best performing channel and corresponding feature per subject
meanperfpersub = df.groupby('sub')[baonly].mean()
meanperfpersubpertype = df.groupby(['type','sub'])[baonly].mean()
meanperfpercohort = df.groupby('cohort')[baonly].mean()
# mean of cohort means
meanofmeans = meanperfpercohort.mean()

idxofmax = np.sort(list(df.groupby(['cohort','sub'])['ba_combined'].idxmax()))
maxpersub = df.iloc[idxofmax][['cohort', 'ba_combined']]

# Plot and save tables
##### Boxplots

### Include normal run for comparison of performances ALL CHANNELS
all =df[['cohort', 'ba_combined']]
allsingle = all.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
allsingle['ba_combined'] = allsingle['ba_combined'].replace('ba_combined','RecursiveFeatureElimination')

dfRef = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")
allRef = dfRef[['cohort', 'ba_combined']]
allmeltedRef = allRef.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
allmeltedRef['ba_combined'] = allmeltedRef['ba_combined'].replace('ba_combined','Reference')

dfSFS = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\SequentialFeatureSelections.csv")
allSFS = dfSFS[['cohort', 'ba_combined']]
allmeltedSFS = allSFS.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
allmeltedSFS['ba_combined'] = allmeltedSFS['ba_combined'].replace('ba_combined','SequentialFeatureSelection')

comparisondf = pd.concat([allmeltedRef,allsingle,allmeltedSFS])
plt.figure()
sns.boxplot(data= comparisondf,x='ba_combined', y='balanced accuracy',hue='cohort')

### Include normal run for comparison of performances BEST CHANNELS
maxsingle = maxpersub.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
maxsingle['ba_combined'] = maxsingle['ba_combined'].replace('ba_combined','RecursiveFeatureElimination')

dfRef = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")
idxofmaxRef = np.sort(list(dfRef.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersubRef = dfRef.iloc[idxofmaxRef][['cohort', 'ba_combined']]
meltedRef = maxpersubRef.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
meltedRef['ba_combined'] = meltedRef['ba_combined'].replace('ba_combined','Reference')

dfSFS = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\SequentialFeatureSelections.csv")
idxofmaxSFS = np.sort(list(dfSFS.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersubSFS = dfSFS.iloc[idxofmaxSFS][['cohort', 'ba_combined']]
meltedSFS = maxpersubSFS.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
meltedSFS['ba_combined'] = meltedSFS['ba_combined'].replace('ba_combined','SequentialFeatureSelection')

comparisondf = pd.concat([meltedRef,maxsingle,meltedSFS])
plt.figure()
sns.boxplot(data= comparisondf,x='ba_combined', y='balanced accuracy',hue='cohort')





######################## Preprocessing
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all.npy"),
    allow_pickle="TRUE",
).item()
featuredim = ch_all['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
features = featuredim[:3] + featuredim[3+1:]
# Dicts are saved as string in DataFrame, so some manipulation to go back to dicts
RFEdictlist = [ast.literal_eval(''.join(x.split('\n')).replace(" ", "").replace("array(","").replace(")","")) for x in df.importance]
SFSdictlist = [ast.literal_eval(''.join(x.split('\n')).replace(" ", "").replace("array(","").replace(")","")) for x in dfSFS.importance]


#### ANALYSIS OVER ALL CHANNELS ##############
# Expand the dataframes
df = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)
df['cv'] = np.resize(np.arange(3),len(df))
for i in range(len(features)):
    name = features[i]
    cv0 = [x['0'][i] for x in RFEdictlist]
    cv1 = [x['1'][i] for x in RFEdictlist]
    cv2 = [x['2'][i] for x in RFEdictlist]
    score = [val for pair in zip(cv0, cv1, cv2) for val in pair]
    df[name] = score

dfSFS = pd.DataFrame(np.repeat(dfSFS.values, 3, axis=0), columns=dfSFS.columns)
dfSFS['cv'] = np.resize(np.arange(3),len(dfSFS))
for i in range(len(features)):
    name = features[i]
    cv0 = [x['0'][i] for x in SFSdictlist]
    cv1 = [x['1'][i] for x in SFSdictlist]
    cv2 = [x['2'][i] for x in SFSdictlist]
    score = [val for pair in zip(cv0, cv1, cv2) for val in pair]
    dfSFS[name] = score
df = df.drop(columns='importance')
dfSFS = dfSFS.drop(columns='importance')

# First naive over all CVs RFE
numfeatselected = df[features].apply(pd.Series.value_counts, axis=1).fillna(0).iloc[:,0]
print(f'Average # of selected features RFE: {np.mean(numfeatselected)}')
plt.figure()
sns.boxplot(numfeatselected)
# Grouped by cohort
datalist = []
concatdf = pd.DataFrame()
for coh in cohorts:
    cohidx = df['cohort'] == coh
    datalist.append(numfeatselected[cohidx])
    concatdf = pd.concat([concatdf, numfeatselected[cohidx]], axis=1)
#sns.boxplot(data= concatdf)
plt.figure()
plt.boxplot(datalist)
plt.xticks(np.arange(4)+1,cohorts)
plt.title('Number of features selected per cohort by Recursive Feature Elimination')

# First naive over all CVs SFS
numfeatselected = dfSFS[features].apply(pd.Series.value_counts, axis=1).fillna(0).iloc[:,1]
print(f'Average # of selected features SFS: {np.mean(numfeatselected)}')
plt.figure()
sns.boxplot(numfeatselected)
# Grouped by cohort
concatdf = pd.DataFrame()
datalist = []
for coh in cohorts:
    cohidx = df['cohort'] == coh
    datalist.append(numfeatselected[cohidx])
    concatdf = pd.concat([concatdf, numfeatselected[cohidx]], axis=1)
#sns.boxplot(data= concatdf)
plt.figure()
plt.boxplot(datalist)
plt.xticks(np.arange(4)+1,cohorts)
plt.title('Number of features selected per cohort by Sequential Feature Selection')

#### Analysis of most selected features
# Naive over all CVs RFE
selected = [df[i].value_counts().iloc[0] for i in features]
selected = np.array(selected) / len(df)
order = np.argsort(selected)[::-1]
plt.figure()
plt.bar(np.array(features)[order],selected[order])
plt.xticks(np.array(features)[order],rotation=45,ha='right')
plt.title('Fraction of runs Recursive Feature Elimination selected the feature (Nested CV; 5 inner)')

# Naive over all CVs SFS
selected = [dfSFS[i].value_counts().iloc[1] if len(dfSFS[i].value_counts()) > 1 else 0 for i in features]
selected = np.array(selected) / len(dfSFS)
order = np.argsort(selected)[::-1]
plt.figure()
plt.bar(np.array(features)[order],selected[order])
plt.xticks(np.array(features)[order],rotation=45,ha='right')
plt.title('Fraction of runs Recursive Sequential Feature Selection selected the feature (Nested CV; 3 inner)')

# per cohort RFE
plt.figure()
count = 0
for coh in cohorts:
    plt.subplot(2, 2, count + 1)
    count += 1
    cohidx = df['cohort'] == coh
    selected = [df[cohidx][i].value_counts().iloc[0] for i in features]
    selected = np.array(selected) / np.sum(cohidx)
    order = np.argsort(selected)
    plt.barh(np.array(features)[order], selected[order])
    plt.yticks(np.array(features)[order], rotation=0, ha='right')
    plt.title(f'{coh} RFE')

# Per cohort SFS
plt.figure()
count = 0
for coh in cohorts:
    plt.subplot(2, 2, count + 1)
    count += 1
    cohidx = dfSFS['cohort'] == coh
    selected = [dfSFS[cohidx][i].value_counts().iloc[1] if len(dfSFS[cohidx][i].value_counts()) > 1 else 0 for i in features]
    selected = np.array(selected) / np.sum(cohidx)
    order = np.argsort(selected)
    plt.barh(np.array(features)[order], selected[order])
    plt.yticks(np.array(features)[order], rotation=0, ha='right')
    plt.title(f'{coh} SFS')









############### ANALYSIS OVER BEST CHANNELS ##############
dfSFS = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\SequentialFeatureSelections.csv")
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\RecursivefeatureeliminationPerformance.csv")
dfSFS = dfSFS.iloc[idxofmaxSFS]
df = df.iloc[idxofmaxRef]
RFEdictlist = [ast.literal_eval(''.join(x.split('\n')).replace(" ", "").replace("array(","").replace(")","")) for x in df.importance]
SFSdictlist = [ast.literal_eval(''.join(x.split('\n')).replace(" ", "").replace("array(","").replace(")","")) for x in dfSFS.importance]
df = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)
df['cv'] = np.resize(np.arange(3),len(df))
for i in range(len(features)):
    name = features[i]
    cv0 = [x['0'][i] for x in RFEdictlist]
    cv1 = [x['1'][i] for x in RFEdictlist]
    cv2 = [x['2'][i] for x in RFEdictlist]
    score = [val for pair in zip(cv0, cv1, cv2) for val in pair]
    df[name] = score

dfSFS = pd.DataFrame(np.repeat(dfSFS.values, 3, axis=0), columns=dfSFS.columns)
dfSFS['cv'] = np.resize(np.arange(3),len(dfSFS))
for i in range(len(features)):
    name = features[i]
    cv0 = [x['0'][i] for x in SFSdictlist]
    cv1 = [x['1'][i] for x in SFSdictlist]
    cv2 = [x['2'][i] for x in SFSdictlist]
    score = [val for pair in zip(cv0, cv1, cv2) for val in pair]
    dfSFS[name] = score

# First naive over all CVs RFE
numfeatselected = df[features].apply(pd.Series.value_counts, axis=1).fillna(0).iloc[:,0]
print(f'Average # of selected features RFE for best channels: {np.mean(numfeatselected)}')
plt.figure()
sns.boxplot(pd.DataFrame(numfeatselected))
# Grouped by cohort
concatdf = pd.DataFrame()
datalist = []
for coh in cohorts:
    cohidx = df['cohort'] == coh
    datalist.append(numfeatselected[cohidx])
    concatdf = pd.concat([concatdf, numfeatselected[cohidx]], axis=1)
#sns.boxplot(data= concatdf)
plt.figure()
plt.boxplot(datalist)
plt.xticks(np.arange(4)+1,cohorts)
plt.title('Number of features selected per cohort by Recursive Feature Elimination')

# First naive over all CVs SFS
numfeatselected = dfSFS[features].apply(pd.Series.value_counts, axis=1).fillna(0).iloc[:,1]
print(f'Average # of selected features SFS: {np.mean(numfeatselected)}')
plt.figure()
sns.boxplot(pd.DataFrame(numfeatselected))
# Grouped by cohort
concatdf = pd.DataFrame()
datalist = []
for coh in cohorts:
    cohidx = df['cohort'] == coh
    datalist.append(numfeatselected[cohidx])
    concatdf = pd.concat([concatdf, numfeatselected[cohidx]], axis=1)
#sns.boxplot(data= concatdf)
plt.figure()
plt.boxplot(datalist)
plt.xticks(np.arange(4)+1,cohorts)
plt.title('Number of features selected per cohort by Sequential Feature Selection')

#### Analysis of most selected features
# Naive over all CVs RFE
selected = [df[i].value_counts().iloc[0] for i in features]
selected = np.array(selected) / len(df)
order = np.argsort(selected)[::-1]
plt.figure()
plt.bar(np.array(features)[order],selected[order])
plt.xticks(np.array(features)[order],rotation=45,ha='right')
plt.title('Fraction of runs Recursive Feature Elimination selected the feature (Nested CV; 5 inner)')

# Naive over all CVs SFS
selected = [dfSFS[i].value_counts().iloc[1] if len(dfSFS[i].value_counts()) > 1 else 0 for i in features]
selected = np.array(selected) / len(dfSFS)
order = np.argsort(selected)[::-1]
plt.figure()
plt.bar(np.array(features)[order],selected[order])
plt.xticks(np.array(features)[order],rotation=45,ha='right')
plt.title('Fraction of runs Recursive Sequential Feature Selection selected the feature (Nested CV; 3 inner)')

# per cohort RFE
plt.figure()
count = 0
for coh in cohorts:
    plt.subplot(2, 2, count + 1)
    count += 1
    cohidx = df['cohort'] == coh
    selected = [df[cohidx][i].value_counts().iloc[0] for i in features]
    selected = np.array(selected) / np.sum(cohidx)
    order = np.argsort(selected)
    plt.barh(np.array(features)[order], selected[order])
    plt.yticks(np.array(features)[order], rotation=0, ha='right')
    plt.title(f'{coh} RFE')

# Per cohort SFS
plt.figure()
count = 0
for coh in cohorts:
    plt.subplot(2, 2, count + 1)
    count += 1
    cohidx = dfSFS['cohort'] == coh
    selected = [dfSFS[cohidx][i].value_counts().iloc[1] if len(dfSFS[cohidx][i].value_counts()) > 1 else 0 for i in features]
    selected = np.array(selected) / np.sum(cohidx)
    order = np.argsort(selected)
    plt.barh(np.array(features)[order], selected[order])
    plt.yticks(np.array(features)[order], rotation=0, ha='right')
    plt.title(f'{coh} SFS')