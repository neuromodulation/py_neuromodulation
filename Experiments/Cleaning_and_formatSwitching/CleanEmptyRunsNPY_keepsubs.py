from sklearn import metrics, model_selection, linear_model, svm, ensemble, kernel_approximation
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import xgboost

# import the data
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

# set features to use (do fft separately, as not every has to be computed)
features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']
performances = []
featuredim = ch_all['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[3] = np.add(idxlist_Berlin_001[3],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))
idxlist_Berlin_001[-1] = np.sort(idxlist_Berlin_001[-1])
idxlist[-1] = np.sort(idxlist[-1])

# TODO: Leave out MedOn for subject 14 of Berlin, the run that does not have movement (due to left arm being used for rotation instead of right)
# TODO: Leave ALL OF sub EL015 --> Also no movement in label for MedOn and MedOff
# TODO: Leave out MedOff for sub EL016 --> No movement label

features = ['combined']
coef = False
del ch_all['Berlin']['EL015']
for cohort in ch_all.keys():
    for sub in ch_all[cohort].keys():
        if cohort == 'Berlin' and sub == '001':
            for channel in ch_all[cohort][sub].keys():
                for featureidx in range(len(features)):
                    for runs in ch_all[cohort][sub][channel].keys():
                        ch_all[cohort][sub][channel][runs]['data'] = ch_all[cohort][sub][channel][runs]['data'][:,idxlist_Berlin_001[5]]
                        ch_all[cohort][sub][channel][runs]['feature_names'] = np.array(ch_all[cohort][sub][channel][runs]['feature_names'])[idxlist_Berlin_001[5]]
        elif cohort == 'Berlin' and sub == '014':
            del ch_all[cohort][sub][list(ch_all[cohort][sub].keys())[-1]]
            for channel in list(ch_all[cohort][sub].keys()):
                for featureidx in range(len(features)):
                    ogruns = list(ch_all[cohort][sub][channel].keys())
                    ch_all[cohort][sub][channel][ogruns[0]]['data'] = ch_all[cohort][sub][channel][ogruns[0]]['data'][:,idxlist[5]]
                    ch_all[cohort][sub][channel][ogruns[0]]['feature_names'] = np.array(ch_all[cohort][sub][channel][ogruns[0]]['feature_names'])[idxlist[5]]
                    if len(ogruns) > 1:
                        del ch_all[cohort][sub][channel][ogruns[1]]
        elif cohort == 'Berlin' and (sub == '004' or sub == '005'):
            ogchan = list(ch_all[cohort][sub].keys())
            for channel in ogchan:
                if np.char.find(channel, '_R_') != -1:
                    chside = 'R_acq'
                else:
                    chside = 'L_acq'
                for featureidx in range(len(features)):
                    runlist = list(ch_all[cohort][sub][channel].keys())
                    for runs in runlist:
                        if np.char.find(runs, chside) != -1: # = on the same side
                            del ch_all[cohort][sub][channel][runs]
                        else:
                            ch_all[cohort][sub][channel][runs]['data'] = ch_all[cohort][sub][channel][runs]['data'][:,idxlist[5]]
                            ch_all[cohort][sub][channel][runs]['feature_names'] = np.array(ch_all[cohort][sub][channel][runs]['feature_names'])[idxlist[5]]
                if not ch_all[cohort][sub][channel]: # Check if now empty, then delete
                    del ch_all[cohort][sub][channel]
        elif cohort == 'Berlin' and sub == 'EL016':
            for channel in list(ch_all[cohort][sub].keys()):
                ogruns = list(ch_all[cohort][sub][channel].keys())
                for runs in ogruns:  # Only include med on (which should be first in the keylist)
                    if np.char.find(runs, 'MedOn') == -1:
                        del ch_all[cohort][sub][channel][runs]
                if not ch_all[cohort][sub][channel]:
                    del ch_all[cohort][sub][channel]
            for channel in list(ch_all[cohort][sub].keys()):
                for featureidx in range(len(features)):
                    for runs in ch_all[cohort][sub][channel].keys():
                        ch_all[cohort][sub][channel][runs]['data'] = ch_all[cohort][sub][channel][runs]['data'][:, idxlist[5]]
                        ch_all[cohort][sub][channel][runs]['feature_names'] = np.array(ch_all[cohort][sub][channel][runs]['feature_names'])[idxlist[5]]
        else:
            for channel in ch_all[cohort][sub].keys():
                for featureidx in range(len(features)):
                    for runs in ch_all[cohort][sub][channel].keys():
                        ch_all[cohort][sub][channel][runs]['data'] = ch_all[cohort][sub][channel][runs]['data'][:,idxlist[5]]
                        ch_all[cohort][sub][channel][runs]['feature_names'] = np.array(ch_all[cohort][sub][channel][runs]['feature_names'])[idxlist[5]]

np.save(r'C:\Users\ICN_GPU\Documents\Glenn_Data\TempCleaned2_channel_all.npy', ch_all)