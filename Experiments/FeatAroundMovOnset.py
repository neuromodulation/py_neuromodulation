# Look into the difference between good vs bad performers in Berlin dataset
import pandas as pd
import numpy as np
from sklearn import tree
import os
import matplotlib.pyplot as plt

############ LOAD in the data ##################
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")

ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")
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

sub_names = []
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
    sub_names.append(sub)
    targets.append(y_concat)

onsetlist = []
for i in range(len(targets)):
    onsetlist.append(np.where((~np.equal(np.array(targets[i]), np.array([False]+list(targets[i][:-1]))) & np.array(targets[i])))[0])

def reject_outliers(data, m=3, repeats = 1):
    for rep in range(repeats):
        med_arr = np.repeat(np.mean(data[abs(data - np.median(data)) < m * np.std(data)]),len(data)) # Take mean without the outlier instead of outlier
        med_arr[abs(data - np.median(data)) < m * np.std(data)] = data[abs(data - np.median(data)) < m * np.std(data)] # Add back original data
    return med_arr

left = 20
right = 20
total = left+right
onsetfeats = np.zeros((len(onsetlist),total,37))
for i in range(len(onsetlist)):
    persub = np.zeros((len(onsetlist[i]),total,37)) # Create empty zeros mat of (nr onsets, features,time around onset)
    for timepoints in range(len(onsetlist[i])): # Get the profile for each sub
        if onsetlist[i][timepoints] >= left and onsetlist[i][timepoints]+right < len(best_ch[i]):
            data = best_ch[i][onsetlist[i][timepoints] - left:onsetlist[i][timepoints] + right, :]
            Hjorthact = data[:,0]
            cleaned = reject_outliers(Hjorthact)
            data[:,0] = cleaned
            persub[timepoints,:,:] = data
    # Remove some outlier for Hjorth activity

    meanforsub = np.mean(persub,0)
    onsetfeats[i,:,:] = meanforsub
plt.figure()
for i in range(np.shape(x_concat)[1]):
    plt.subplot(6, 7, i + 1)
    #for subs in range(len(onsetfeats)):
        #plt.scatter(range(total),onsetfeats[subs,:,i],s=1)
    plt.plot(np.mean(onsetfeats[:,:,i],0))
    plt.fill_between(range(total), np.mean(onsetfeats[:,:,i],0) - np.std(onsetfeats[:,:,i],0), np.mean(onsetfeats[:,:,i],0) + np.std(onsetfeats[:,:,i],0), alpha=.1)
    plt.xticks([0,left,total-1],[-left/10,0,right/10])
    plt.title(f'{featuredim[i]}')
plt.suptitle('Features around movement onset')
plt.show()

plt.figure()
plt.imshow(np.mean(onsetfeats[:,:,:],0).T)
plt.colorbar()
plt.xticks([0,left,total-1],[-left/10,0,right/10])
plt.xlabel('Time [s]')
plt.yticks(range(37),featuredim)
plt.title('Features around movement onset')
plt.show()
