# Look into the difference between good vs bad performers in Berlin dataset
import pandas as pd
import numpy as np
from sklearn import tree
import os
import matplotlib.pyplot as plt

############ LOAD in the data ##################
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")

ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_noraw.npy"),
    allow_pickle="TRUE",
).item()

################### Find the best channel per subject ##################################
idxofmax = np.sort(list(df.groupby(['sub','cohort'])['ba_combined'].idxmax()))
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
Beijingdata = [best_ch[i] for i in np.where(cohsubchmax['cohort'].values == 'Beijing')[0]]
Pittdata = [best_ch[i] for i in np.where(cohsubchmax['cohort'].values == 'Pittsburgh')[0]]
Washdata = [best_ch[i] for i in np.where(cohsubchmax['cohort'].values == 'Washington')[0]]

# Try to find the optimal offset such that the sample will be negative
meanmovlen = []
meantimebetween = []
nrdiscrmov = []
stdmovlen = []
movlen = []
timebetween = []
stdtimebetween = []
movlen_flat = []
timebetw_flat = []
for i in range(len(targets)):
    m = targets[i] != (np.r_[False, targets[i][:-1]]) # Find the moments where a switch between False True is happening
    _, c = np.unique(m.cumsum(), return_index=True) # Give index of each True statement (switch between True/False)
    if not bool(targets[i][0]):
        c_act = c[1:] # Remove the first index which is always included in c (first unique value)
        out = c_act[1::2] - c_act[::2]
        betw = np.append(c_act[::2], [len(targets[i])], ) - np.append([0], c_act[1::2])
    else:
        out = c[1::2] - c[::2] # Except when the first index is actually True/1
        betw = np.append(c[::2], [len(targets[i])], ) - np.append([0], c[1::2])
    nrdiscrmov.append(len(out))
    # movement
    meanmovlen.append(np.mean(out))
    movlen.append(out)
    stdmovlen.append(np.std(out))
    movlen_flat.append(out)
    if any(betw>800):
        print(i)
        print(betw)
    timebetween.append(betw)
    meantimebetween.append(np.mean(betw))
    stdtimebetween.append(np.std(betw))
    timebetw_flat.append(betw)
movlen_flat = np.concatenate(movlen_flat,axis=None)
timebetw_flat = np.concatenate(timebetw_flat,axis=None)
plt.figure()
plt.boxplot(meanmovlen)
plt.title('mean movement duration per subject (samples)')
plt.show()
plt.figure()
plt.boxplot(meantimebetween)
plt.title('mean time between mov per subject (samples)')
plt.show()
plt.figure()
plt.boxplot(movlen)
plt.title('movement duration for per subjects (samples)')
plt.show()
plt.figure()
plt.boxplot(timebetween ,showfliers=False)
plt.title('time between movement for per subjects (samples)')
plt.show()


plt.figure()
plt.boxplot(movlen_flat)
plt.title('movement duration for all subjects (samples)')
plt.show()
plt.figure()
plt.boxplot(timebetw_flat,showfliers=False)
plt.title('time between movement for all subjects (samples)')
plt.show()

plt.figure()
movlen_percentile = []
percentiles = np.linspace(0,100,1000)
for i in percentiles:
    movlen_percentile.append(np.percentile(movlen_flat,i))
plt.plot(percentiles,movlen_percentile)
plt.xlabel('Percentile')
plt.ylabel('Movement duration')

# Let say we are satisfied with between small majority and 70% coverage of movement in 80% of cases (example)
# Receptive field is x --> Need mov duration >0.7*x --> percentile of this is 0.2 (find)
# So x can be 1/0.7 * mov_dur@20th percentile
coverage = 0.7 # >70% of the receptive field covered by movement
certainty = 0.75 # >75% of the cases
receptive_field = movlen_percentile[int((100-certainty*100)*10)]*1/coverage
lowerboundcov = 0.9
lowerbound = receptive_field*lowerboundcov
lowerbpercentile = np.argmin(abs(movlen_percentile-lowerbound))
print(f'The receptive field size to be to covered up to {coverage*100}% movement in >{certainty*100}% of cases is: {receptive_field}')
print(f'You will get maximum coverage of {lowerboundcov*100}% in {100-lowerbpercentile/10}% of cases')

# Actually will always get majority movement with unbalanced receptive field --> Until (RF-1)/2>movlen
Maj = 0.9 # Always get majority in 90% of cases
RF = movlen_percentile[int((100-Maj*100)*10)]*2+1
print(f'Receptive field to keep a majority of RF movement in {Maj*100}% of cases: {RF}')