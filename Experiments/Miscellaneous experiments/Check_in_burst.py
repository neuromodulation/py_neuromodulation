# Look into the difference between good vs bad performers in Berlin dataset
import pandas as pd
import numpy as np
from sklearn import tree
import os
import matplotlib.pyplot as plt
from itertools import groupby

############ LOAD in the data ##################
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\AllfeaturePerformances_TempCleaned.csv")

ch_all_fft = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

ch_all_feat = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "TempCleaned2_channel_all.npy"),
    allow_pickle="TRUE",
).item()

updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")
############# Specify features to create the feature indexlist to exclude the fooof knee from analysis ################
features = ['Hjorth','raw','fft','Sharpwave', 'fooof', 'bursts', 'combined']
performances = []
featuredim = ch_all_feat['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
featuredim001 = ch_all_feat['Berlin']['001']['ECOG_L_1_2_SMC_AT-avgref']['sub-001_ses-EcogLfpMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg']['feature_names']

idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[5] = np.add(idxlist_Berlin_001[5],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))

featuredim = featuredim[idxlist[-1]]

################### Find the best channel per subject ##################################
idxofmax = np.sort(list(df.groupby(['cohort','sub'])['ba_combined'].idxmax()))
cohsubchmax = df.iloc[idxofmax][['cohort','sub','ch']]
ba_combined = df.iloc[idxofmax]['ba_combined']

sub_names = []
cohs = []
best_ch = []
targets = [] # For analysis of the labels for the best channels --> length / movement samples etc.
for i in range(len(cohsubchmax)):
    cohort = cohsubchmax['cohort'].values[i]
    sub = cohsubchmax['sub'].values[i]
    channel = cohsubchmax['ch'].values[i]
    x_concat = []
    y_concat = []
    for runs in ch_all_feat[cohort][sub][channel].keys():
        x_concat.append(np.squeeze(ch_all_feat[cohort][sub][channel][runs]['data'][:, idxlist[-1]]))
        y_concat.append(ch_all_feat[cohort][sub][channel][runs]['label'])
    x_concat = np.concatenate(x_concat, axis=0)
    y_concat = np.concatenate(y_concat, axis=0)
    cohs.append(cohort)
    best_ch.append(x_concat)
    sub_names.append(sub)
    targets.append(y_concat)
duppersub = []
valduppersub = []
subhasstuck = []
subcount = 0
for cursub in best_ch:
    dupsperfeat = []
    numdupsperfeat = []
    valdupperfeat = []
    stuck = 0
    for feat in range(36):
        count_dups = [sum(1 for _ in group) for _, group in groupby(list(cursub[:,feat]))]
        val_dups = [val for val, _ in groupby(list(cursub[:,feat]))]
        dupsperfeat.append(np.array(count_dups)[np.logical_and(np.array(count_dups) > 500,np.logical_and(np.array(val_dups) >-3+1e-5,np.array(val_dups) < 3-1e-5))])
        if len(dupsperfeat[feat]) >= 1:
            stuck = 1
        valdupperfeat.append(np.array(val_dups)[np.logical_and(np.array(count_dups) > 500,np.logical_and(np.array(val_dups) >-3+1e-5,np.array(val_dups) < 3-1e-5))])
    if stuck:
        subhasstuck.append(subcount)
    print(subcount)
    subcount +=1
    duppersub.append(dupsperfeat)
    valduppersub.append(valdupperfeat)
