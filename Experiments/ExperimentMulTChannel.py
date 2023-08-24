from sklearn import metrics, model_selection, linear_model
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

# import the data
ch_all = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
    allow_pickle="TRUE",
).item()
df = pd.read_csv(r"D:\Glenn\df_all_features.csv")

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

kf = KFold(n_splits = 3, shuffle = False)
model = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
# loop over all channels --> run per subject model for best x channels concat (as features)
performancedict = {}
for cohort in ch_all.keys():
    print(cohort)
    performancedict[cohort] = {}
    for sub in ch_all[cohort].keys():
        print(sub)
        performancedict[cohort][sub] = {}
        ch_test_list = \
        sortedChannels = df.query(f"cohort == @cohort and sub == @sub").sort_values(
            by='ba_combined', ascending=False)[
            "ch"]
        for nrchannels in range(1,5):
            chlist = sortedChannels.values[:nrchannels]
            performancedict[cohort][sub][str(nrchannels)] = {}
            x_concat = []
            y_concat = []
            for i in range(nrchannels): # concat as features
                if not (cohort == 'Berlin' and sub == 'EL015'):
                    channel = chlist[i]
                x_concat_per = []
                y_concat_per = []
                if cohort == 'Berlin' and sub == '001':
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat_per.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist_Berlin_001[5]]))
                        y_concat_per.append(ch_all[cohort][sub][channel][runs]['label'])
                elif cohort == 'Berlin' and sub == '014':
                    for runs in ch_all[cohort][sub][channel].keys()[0]:
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[5]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                elif cohort == 'Berlin' and sub == 'EL015':
                    break
                elif cohort == 'Berlin' and sub == 'EL016':
                    for runs in ch_all[cohort][sub][
                        channel].keys():  # Only include med on (which should be first in the keylist)
                        if np.char.find(runs, 'MedOn') != -1:
                            x_concat.append(
                                np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:, idxlist[5]]))
                            y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                        else:
                            continue
                else:
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat_per.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[5]]))
                        y_concat_per.append(ch_all[cohort][sub][channel][runs]['label'])
                x_concat_per = np.concatenate(x_concat_per, axis=0)
                y_concat_per = np.concatenate(y_concat_per, axis=0)
                x_concat.append(x_concat_per)
            x_concat = np.concatenate(x_concat, axis=1)
            scores = cross_val_score(model, x_concat, np.array(y_concat_per, dtype=int), cv=kf, scoring = bascorer)
            performancedict[cohort][sub][str(nrchannels)]['ba'] = np.mean(scores)
            print(np.mean(scores))