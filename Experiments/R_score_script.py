from sklearn import metrics, model_selection, linear_model
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

# import the data
ch_all = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

# set features to use (do fft separately, as not every has to be computed)
features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft']
performances = []
featuredim = ch_all['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
idxlist = []
for i in range(len(features)):
    idx_i = list(np.nonzero(np.char.find(featuredim, features[i])+1))
    idxlist.append(idx_i)
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[3] = np.add(idxlist_Berlin_001[3],1)


kf = KFold(n_splits = 3, shuffle = False)
model = linear_model.LogisticRegression(class_weight="balanced", max_iter=500)
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
# loop over all channels
performancedict = {}
for cohort in ch_all.keys():
    print(cohort)
    performancedict[cohort] = {}
    for sub in [list(ch_all[cohort].keys())[13]]: # TODO: Fix for subject 14: y_pred != y_true classes --> Prob due to no movement present in split --> Subject has few movement ?
        print(sub)
        performancedict[cohort][sub] = {}

        if cohort == 'Berlin' and sub == '001':
            for channel in ch_all[cohort][sub].keys():
                performancedict[cohort][sub][channel] = {}
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist_Berlin_001[featureidx]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                    x_concat = np.concatenate(x_concat, axis=0)
                    y_concat = np.concatenate(y_concat, axis=0)
                    scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer)
                    performancedict[cohort][sub][channel][features[featureidx]] = np.mean(scores)

        else:
            for channel in ch_all[cohort][sub].keys():
                performancedict[cohort][sub][channel] = {}
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[featureidx]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                    x_concat = np.concatenate(x_concat,axis=0)
                    y_concat = np.concatenate(y_concat,axis=0)
                    print(np.unique(np.array(y_concat,dtype=int)))
                    print(np.shape(x_concat))
                    scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer)
                    performancedict[cohort][sub][channel][features[featureidx]] = np.mean(scores)

