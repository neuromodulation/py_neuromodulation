# Look into the difference between good vs bad performers in Berlin dataset
import pandas as pd
import numpy as np
from sklearn import tree
import os
import matplotlib.pyplot as plt

############ LOAD in the data ##################
df = pd.read_csv(r"D:\Glenn\df_all_features.csv")

ch_all = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
    allow_pickle="TRUE",
).item()
############# Specify features to create the feature indexlist to exclude the fooof knee from analysis ################
features = ['Hjorth','raw','fft','Sharpwave', 'fooof', 'bursts', 'combined']
performances = []
featuredim = ch_all['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
featuredim001 = ch_all['Berlin']['001']['ECOG_L_1_2_SMC_AT-avgref']['sub-001_ses-EcogLfpMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg']['feature_names']

idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[5] = np.add(idxlist_Berlin_001[5],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))

################### Find the best channel per subject ##################################
idxofmax = list(df.groupby('sub')['ba_combined'].idxmax())
cohsubchmax = df.iloc[idxofmax][['cohort','sub','ch']]
ba_combined = df.iloc[idxofmax]['ba_combined']

best_ch = []
targets = [] # For analysis of the labels for the best channels --> length / movement samples etc.
for i in range(len(cohsubchmax)):
    cohort = cohsubchmax['cohort'].values[i]
    sub = cohsubchmax['sub'].values[i]
    channel = cohsubchmax['ch'].values[i]
    x_concat = []
    y_concat = []
    for runs in ch_all[cohort][sub][channel].keys():
        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data']))
        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
    x_concat = np.concatenate(x_concat, axis=0)
    y_concat = np.concatenate(y_concat, axis=0)
    best_ch.append(x_concat)
    targets.append(y_concat)

#####################  Get the bottom x percent and the top x percent subjects from Berlin #####################
Berlindata = [best_ch[i] for i in np.where(cohsubchmax['cohort'].values == 'Berlin')[0]]
Berlintargets = [targets[i] for i in np.where(cohsubchmax['cohort'].values == 'Berlin')[0]]
Berlinba = list(ba_combined.iloc[np.where(cohsubchmax['cohort'].values == 'Berlin')[0]])
lowperf = np.percentile(Berlinba,25,method='lower')
highperf = np.percentile(Berlinba,75,method='lower')
Berlinlow = [Berlindata[i] for i in np.where(Berlinba<=lowperf)[0]]
Berlinhigh = [Berlindata[i] for i in np.where(Berlinba>=highperf)[0]]
targlow = [Berlintargets[i] for i in np.where(Berlinba<=lowperf)[0]]
targhigh = [Berlintargets[i] for i in np.where(Berlinba>=highperf)[0]]

###################### Analyze the movement information for differences ####################
nrmovlow = [np.sum(i) for i in targlow]
nrmovhigh = [np.sum(i) for i in targhigh]
nrrestlow = [len(targlow[i])-nrmovlow[i] for i in range(len(targlow))]
nrresthigh = [len(targhigh[i])-nrmovhigh[i] for i in range(len(targhigh))]

# Boxplot of the number of movements for best and worst performing subjects
plt.boxplot([nrmovlow,nrmovhigh])
plt.show()

plt.boxplot([nrrestlow,nrresthigh])
plt.show()

meanmovlenlow = []
nrdiscrmovlow = []
stdmovlenlow = []
for i in range(len(targlow)):
    m = targlow[i] != (np.r_[np.nan, targlow[i][:-1]])
    _, c = np.unique(m.cumsum(), return_index=True)
    out = np.diff(np.r_[c, len(targlow[i])])
    nrdiscrmovlow.append(len(c))
    meanmovlenlow.append(np.mean(out))
    stdmovlenlow.append(np.std(out))
nrdiscrmovhigh = []
meanmovlenhigh = []
stdmovlenhigh = []
for i in range(len(targhigh)):
    m = targhigh[i] != (np.r_[np.nan, targhigh[i][:-1]])
    _, c = np.unique(m.cumsum(), return_index=True)
    nrdiscrmovhigh.append(len(c))
    out = np.diff(np.r_[c, len(targhigh[i])])
    meanmovlenhigh.append(np.mean(out))
    stdmovlenhigh.append(np.std(out))
# Boxplot of the mean movement lengths of best and lowest subjects + standard deviation in movement length (might be proxy for EMG quality)
plt.boxplot([nrdiscrmovlow,nrdiscrmovhigh])
plt.show()
plt.boxplot([meanmovlenlow,meanmovlenhigh])
plt.show()
plt.boxplot([stdmovlenlow,stdmovlenhigh])
plt.show()

################## Analysis of features differences (mean, std over all time / during movement & during rest) ###########################
# All time
stdperfeatlow = []
meanperfeatlow = []
for i in range(len(Berlinlow)):
    stdperfeatlow.append(np.std(Berlinlow[i],axis=0))
    meanperfeatlow.append(np.mean(Berlinlow[i],axis=0))
stdperfeatlow[0] = stdperfeatlow[0][idxlist_Berlin_001[6]]
stdperfeathigh = []
meanperfeathigh = []
for i in range(len(Berlinhigh)):
    stdperfeathigh.append(np.std(Berlinhigh[i],axis=0))
    meanperfeathigh.append(np.mean(Berlinhigh[i], axis=0))
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6,7,i+1)
    plt.boxplot([[item[i] for item in stdperfeatlow],[item[i] for item in stdperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Standard deviation in features of low vs high performers')
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6, 7, i + 1)
    plt.boxplot([[item[i] for item in meanperfeatlow],[item[i] for item in meanperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Mean of features of low vs high performers')

# Variation in recording signal during movement
stdperfeatlow = []
meanperfeatlow = []
for i in range(len(Berlinlow)):
    stdperfeatlow.append(np.std(Berlinlow[i][np.where(targlow[i])],axis=0))
    meanperfeatlow.append(np.mean(Berlinlow[i][np.where(targlow[i])],axis=0))
stdperfeatlow[0] = stdperfeatlow[0][idxlist_Berlin_001[6]]
stdperfeathigh = []
meanperfeathigh = []
for i in range(len(Berlinhigh)):
    stdperfeathigh.append(np.std(Berlinhigh[i][np.where(targhigh[i])],axis=0))
    meanperfeathigh.append(np.mean(Berlinhigh[i][np.where(targhigh[i])], axis=0))
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6,7,i+1)
    plt.boxplot([[item[i] for item in stdperfeatlow],[item[i] for item in stdperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Standard deviation in features during movement of low vs high performers')
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6, 7, i + 1)
    plt.boxplot([[item[i] for item in meanperfeatlow],[item[i] for item in meanperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Mean of features during movement of low vs high performers')

# Variation in recording signal during rest
stdperfeatlow = []
meanperfeatlow = []
for i in range(len(Berlinlow)):
    stdperfeatlow.append(np.std(Berlinlow[i][~np.where(targlow[i])[0]],axis=0))
    meanperfeatlow.append(np.mean(Berlinlow[i][~np.where(targlow[i])[0]],axis=0))
stdperfeatlow[0] = stdperfeatlow[0][idxlist_Berlin_001[6]]
stdperfeathigh = []
meanperfeathigh = []
for i in range(len(Berlinhigh)):
    stdperfeathigh.append(np.std(Berlinhigh[i][~np.where(targhigh[i])[0]],axis=0))
    meanperfeathigh.append(np.mean(Berlinhigh[i][~np.where(targhigh[i])[0]], axis=0))
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6,7,i+1)
    plt.boxplot([[item[i] for item in stdperfeatlow],[item[i] for item in stdperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Standard deviation in rest  of low vs high performers')
plt.figure()
for i in range(len(stdperfeathigh[0])):
    plt.subplot(6, 7, i + 1)
    plt.boxplot([[item[i] for item in meanperfeatlow],[item[i] for item in meanperfeathigh]])
    plt.title(f'{featuredim[i]}')
    plt.suptitle('Mean of features during rest of low vs high performers')


################### Compare their movement related features --> Mean feature profile around movement ################


### Second part: Train a model to predict performance output (all best channels) and check what features it looks for (if the model works)
tree =  tree.DecisionTreeRegressor()
tree.fit(best_ch,ba_combined)

################## Findings: ########################
# From comparing berlin it seems that there is MORE beta suppression for the best performing subjects.
# --> This seems to correlate to UPDRS

# TODO: Potentially use an interpretable RNN to see what features predict the balanced accuracy score of a channel

# TODO: Different between movement and rest feature values per subject --> See Cohort effect