# Look into the difference between good vs bad performers in Berlin dataset
import pandas as pd
import numpy as np
from sklearn import tree
import os
import matplotlib.pyplot as plt

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

onsetlist = []
for i in range(len(targets)):
    onsetlist.append(np.where((~np.equal(np.array(targets[i]), np.array([False]+list(targets[i][:-1]))) & np.array(targets[i])))[0])

def reject_outliers(data, m=3, repeats = 1):
    for rep in range(repeats):
        med_arr = np.repeat(np.mean(data[abs(data - np.median(data)) < m * np.std(data)+1e-5]),len(data)) # Take mean without the outlier instead of outlier
        med_arr[abs(data - np.median(data)) < m * np.std(data)+1e-5] = data[abs(data - np.median(data)) < m * np.std(data)+1e-5] # Add back original data
    return med_arr

left = 20
right = 20
total = left+right
onsetfeats = np.zeros((len(onsetlist),total,36))
for i in range(len(onsetlist)):
    persub = np.zeros((len(onsetlist[i]),total,36)) # Create empty zeros mat of (nr onsets, features,time around onset)
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
cbar = plt.colorbar()
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=15)
cbar.set_label('Amplitude [a.u.]',fontsize=18)
plt.xticks([0,left,total-1],[-left/10,0,right/10],fontsize=20)
plt.xlabel('Time aligned to movement [s]', fontsize=25)
plt.yticks(range(36),featuredim,fontsize=10)
plt.title('Feature estimation ',fontsize=18)
plt.show()
plt.savefig(r"C:\Users\ICN_GPU\Documents\Glenn_Data\Figures\PosterFigures\HeatmapFeaturesAroundMovementOnset.pdf")

############# Analyze Berlin vs rest ##############
# And plot the cohorts in the same figure
samefig, sameax = plt.subplots(2,2)
from mpl_toolkits.axes_grid1 import make_axes_locatable

Berlinbool = [cohs[i] == 'Berlin' for i in range(len(cohs))]
onsetfeatBer = onsetfeats[Berlinbool]
plt.figure()
for i in range(np.shape(x_concat)[1]):
    plt.subplot(6, 7, i + 1)
    #for subs in range(len(onsetfeats)):
        #plt.scatter(range(total),onsetfeats[subs,:,i],s=1)
    plt.plot(np.mean(onsetfeatBer[:,:,i],0))
    plt.fill_between(range(total), np.mean(onsetfeatBer[:,:,i],0) - np.std(onsetfeatBer[:,:,i],0), np.mean(onsetfeatBer[:,:,i],0) + np.std(onsetfeatBer[:,:,i],0), alpha=.1)
    plt.xticks([0,left,total-1],[-left/10,0,right/10])
    plt.title(f'{featuredim[i]}')
plt.suptitle('Features around movement onset')
plt.show()

im1 = sameax[0,0].imshow(np.mean(onsetfeatBer[:,:,:],0).T)
divider = make_axes_locatable(sameax[0,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_label('Amplitude [a.u.]')
samefig.colorbar(im1, cax=cax, orientation='vertical',label='Amplitude [a.u.]')
ticklabs = cax.get_yticklabels()
sameax[0,0].set_xticks([0,left,total-1],[-left/10,0,right/10])
sameax[0,0].set_xlabel('Time aligned to movement [s]')
sameax[0,0].set_yticks(range(36),featuredim)
sameax[0,0].set_title('Berlin')

# Berlin per subject
Berlinbool = [cohs[i] == 'Berlin' for i in range(len(cohs))]
onsetfeatBer = onsetfeats[Berlinbool]
idxTrue = [i for i in range(len(Berlinbool)) if Berlinbool[i]]
plt.figure()
for j in range(sum(Berlinbool)):
    plt.subplot(3, 6, j + 1)
    cursub = idxTrue[j]
    plt.imshow(onsetfeats[cursub,:,:].T)
    cbar = plt.colorbar()
    plt.xticks([0, left, total - 1], [-left / 10, 0, right / 10])
    plt.title(f'sub {j}, R {ba_combined.iloc[cursub]:.2f}')
    plt.show()

## Pittsburgh
Pittbool = [cohs[i] == 'Pittsburgh' for i in range(len(cohs))]
onsetfeatPitt = onsetfeats[Pittbool]
plt.figure()
for i in range(np.shape(x_concat)[1]):
    plt.subplot(6, 7, i + 1)
    #for subs in range(len(onsetfeats)):
        #plt.scatter(range(total),onsetfeats[subs,:,i],s=1)
    plt.plot(np.mean(onsetfeatPitt[:,:,i],0))
    plt.fill_between(range(total), np.mean(onsetfeatPitt[:,:,i],0) - np.std(onsetfeatPitt[:,:,i],0), np.mean(onsetfeatPitt[:,:,i],0) + np.std(onsetfeatPitt[:,:,i],0), alpha=.1)
    plt.xticks([0,left,total-1],[-left/10,0,right/10])
    plt.title(f'{featuredim[i]}')
plt.suptitle('Features around movement onset')
plt.show()

im2 = sameax[0,1].imshow(np.mean(onsetfeatPitt[:,:,:],0).T)
divider = make_axes_locatable(sameax[0,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
samefig.colorbar(im2, cax=cax, orientation='vertical',label='Amplitude [a.u.]')
ticklabs = cax.get_yticklabels()
sameax[0,1].set_xticks([0,left,total-1],[-left/10,0,right/10])
sameax[0,1].set_xlabel('Time aligned to movement [s]')
sameax[0,1].set_yticks(range(36),'')
sameax[0,1].set_title('Pittsburgh')

# Pitt per subject
idxTrue = [i for i in range(len(Pittbool)) if Pittbool[i]]
plt.figure()
for j in range(sum(Pittbool)):
    plt.subplot(3, 6, j + 1)
    cursub = idxTrue[j]
    plt.imshow(onsetfeats[cursub,:,:].T)
    cbar = plt.colorbar()
    plt.xticks([0, left, total - 1], [-left / 10, 0, right / 10])
    plt.title(f'sub {j}, R {ba_combined.iloc[cursub]:.2f}')
    plt.show()

## Beijing
Beibool = [cohs[i] == 'Beijing' for i in range(len(cohs))]
onsetfeatBei = onsetfeats[Beibool]
plt.figure()
for i in range(np.shape(x_concat)[1]):
    plt.subplot(6, 7, i + 1)
    #for subs in range(len(onsetfeats)):
        #plt.scatter(range(total),onsetfeats[subs,:,i],s=1)
    plt.plot(np.mean(onsetfeatBei[:,:,i],0))
    plt.fill_between(range(total), np.mean(onsetfeatBei[:,:,i],0) - np.std(onsetfeatBei[:,:,i],0), np.mean(onsetfeatBei[:,:,i],0) + np.std(onsetfeatBei[:,:,i],0), alpha=.1)
    plt.xticks([0,left,total-1],[-left/10,0,right/10])
    plt.title(f'{featuredim[i]}')
plt.suptitle('Features around movement onset')
plt.show()

im3 = sameax[1,0].imshow(np.mean(onsetfeatBei[:,:,:],0).T)
divider = make_axes_locatable(sameax[1,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
samefig.colorbar(im1, cax=cax, orientation='vertical',label='Amplitude [a.u.]')
ticklabs = cax.get_yticklabels()
#cax.set_yticklabels(ticklabs, fontsize=15)
#cax.set_label('Amplitude [a.u.]',fontsize=18)
sameax[1,0].set_xticks([0,left,total-1],[-left/10,0,right/10])
sameax[1,0].set_xlabel('Time aligned to movement [s]')
sameax[1,0].set_yticks(range(36),featuredim)
sameax[1,0].set_title('Beijing')

# Bei per subject
idxTrue = [i for i in range(len(Beibool)) if Beibool[i]]
plt.figure()
for j in range(sum(Beibool)):
    plt.subplot(3, 6, j + 1)
    cursub = idxTrue[j]
    plt.imshow(onsetfeats[cursub,:,:].T)
    cbar = plt.colorbar()
    plt.xticks([0, left, total - 1], [-left / 10, 0, right / 10])
    plt.title(f'sub {j}, R {ba_combined.iloc[cursub]:.2f}')
    plt.show()

# Washington
Washbool = [cohs[i] == 'Washington' for i in range(len(cohs))]
onsetfeatWash = onsetfeats[Washbool]
plt.figure()
for i in range(np.shape(x_concat)[1]):
    plt.subplot(6, 7, i + 1)
    #for subs in range(len(onsetfeats)):
        #plt.scatter(range(total),onsetfeats[subs,:,i],s=1)
    plt.plot(np.mean(onsetfeatWash[:,:,i],0))
    plt.fill_between(range(total), np.mean(onsetfeatWash[:,:,i],0) - np.std(onsetfeatWash[:,:,i],0), np.mean(onsetfeatWash[:,:,i],0) + np.std(onsetfeatWash[:,:,i],0), alpha=.1)
    plt.xticks([0,left,total-1],[-left/10,0,right/10])
    plt.title(f'{featuredim[i]}')
plt.suptitle('Features around movement onset')
plt.show()

im4 = sameax[1,1].imshow(np.mean(onsetfeatWash[:,:,:],0).T)
divider = make_axes_locatable(sameax[1,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
samefig.colorbar(im1, cax=cax, orientation='vertical',label='Amplitude [a.u.]')
ticklabs = cax.get_yticklabels()
#cax.set_yticklabels(ticklabs, fontsize=15)
#cax.set_label('Amplitude [a.u.]',fontsize=18)
sameax[1,1].set_xticks([0,left,total-1],[-left/10,0,right/10])
sameax[1,1].set_xlabel('Time aligned to movement [s]')
sameax[1,1].set_yticks(range(36),'')
sameax[1,1].set_title('Washington')
samefig.suptitle('Features aligned to movement onset for the different cohorts')