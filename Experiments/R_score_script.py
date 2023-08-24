from sklearn import metrics, model_selection, linear_model
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate

# import the data
ch_all = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
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

kf = KFold(n_splits = 3, shuffle = False)
model = linear_model.LogisticRegression(solver='liblinear',penalty='l1',class_weight="balanced", max_iter=1000)
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
# loop over all channels
performancedict = {}
coefdict = {}
for cohort in ch_all.keys():
    print(cohort)
    performancedict[cohort] = {}
    coefdict[cohort] = {}
    for sub in ch_all[cohort].keys():
        print(sub)
        performancedict[cohort][sub] = {}
        coefdict[cohort][sub] = {}

        if cohort == 'Berlin' and sub == '001':
            for channel in ch_all[cohort][sub].keys():
                performancedict[cohort][sub][channel] = {}
                performancedict[cohort][sub][channel]['ba'] = {}
                performancedict[cohort][sub][channel]['95%CI'] = {}
                coefdict[cohort][sub][channel] = {}

                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist_Berlin_001[featureidx]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                    x_concat = np.concatenate(x_concat, axis=0)
                    y_concat = np.concatenate(y_concat, axis=0)
                    cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                    scores = cv_out['test_score']
                    allcoefs = []
                    for model in cv_out['estimator']:
                        allcoefs.append(model.coef_)
                    avgcoeff = np.mean(allcoefs,axis=0)
                    coefdict[cohort][sub][channel][features[featureidx]] = avgcoeff
                    performancedict[cohort][sub][channel]['ba'][features[featureidx]] = np.mean(scores)
                    performancedict[cohort][sub][channel]['95%CI'][features[featureidx]] = np.std(scores)*2
                performancedict[cohort][sub][channel]['explength'] = len(y_concat)
                performancedict[cohort][sub][channel]['movsamples'] = np.sum(y_concat)
        elif cohort == 'Berlin' and sub == '014':
            for channel in list(ch_all[cohort][sub].keys())[:-1]:
                performancedict[cohort][sub][channel] = {}
                performancedict[cohort][sub][channel]['ba'] = {}
                performancedict[cohort][sub][channel]['95%CI'] = {}

                coefdict[cohort][sub][channel] = {}
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in [list(ch_all[cohort][sub][channel].keys())[0]]: # Only include med on (which should be first in the keylist)
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[featureidx]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                    x_concat = np.concatenate(x_concat, axis=0)
                    y_concat = np.concatenate(y_concat, axis=0)
                    cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                    scores = cv_out['test_score']
                    allcoefs = []
                    for model in cv_out['estimator']:
                        allcoefs.append(model.coef_)
                    avgcoeff = np.mean(allcoefs,axis=0)
                    coefdict[cohort][sub][channel][features[featureidx]] = avgcoeff
                    performancedict[cohort][sub][channel]['ba'][features[featureidx]] = np.mean(scores)
                    performancedict[cohort][sub][channel]['95%CI'][features[featureidx]] = np.std(scores)*2
                performancedict[cohort][sub][channel]['explength'] = len(y_concat)
                performancedict[cohort][sub][channel]['movsamples'] = np.sum(y_concat)
        elif cohort == 'Berlin' and sub == 'EL015':
            del coefdict[cohort][sub]
            del performancedict[cohort][sub]
        elif cohort == 'Berlin' and sub == 'EL016':
            for channel in list(ch_all[cohort][sub].keys()):
                performancedict[cohort][sub][channel] = {}
                performancedict[cohort][sub][channel]['ba'] = {}
                performancedict[cohort][sub][channel]['95%CI'] = {}

                coefdict[cohort][sub][channel] = {}
                try: # Just to get around the one channel that does not have MedOn
                    for featureidx in range(len(features)):
                        x_concat = []
                        y_concat = []
                        for runs in ch_all[cohort][sub][channel].keys(): # Only include med on (which should be first in the keylist)
                            if np.char.find(runs, 'MedOn') != -1:
                                x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[featureidx]]))
                                y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                            else:
                                continue
                        x_concat = np.concatenate(x_concat, axis=0)
                        y_concat = np.concatenate(y_concat, axis=0)
                        cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                        scores = cv_out['test_score']
                        allcoefs = []
                        for model in cv_out['estimator']:
                            allcoefs.append(model.coef_)
                        avgcoeff = np.mean(allcoefs, axis=0)
                        coefdict[cohort][sub][channel][features[featureidx]] = avgcoeff
                        performancedict[cohort][sub][channel]['ba'][features[featureidx]] = np.mean(scores)
                        performancedict[cohort][sub][channel]['95%CI'][features[featureidx]] = np.std(scores)*2
                    performancedict[cohort][sub][channel]['explength'] = len(y_concat)
                    performancedict[cohort][sub][channel]['movsamples'] = np.sum(y_concat)
                except:
                    del performancedict[cohort][sub][channel]
        else:
            for channel in ch_all[cohort][sub].keys():
                performancedict[cohort][sub][channel] = {}
                performancedict[cohort][sub][channel]['ba'] = {}
                performancedict[cohort][sub][channel]['95%CI'] = {}

                coefdict[cohort][sub][channel] = {}
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in ch_all[cohort][sub][channel].keys():
                        x_concat.append(np.squeeze(ch_all[cohort][sub][channel][runs]['data'][:,idxlist[featureidx]]))
                        y_concat.append(ch_all[cohort][sub][channel][runs]['label'])
                    x_concat = np.concatenate(x_concat,axis=0)
                    y_concat = np.concatenate(y_concat,axis=0)
                    cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                    scores = cv_out['test_score']
                    allcoefs = []
                    for model in cv_out['estimator']:
                        allcoefs.append(model.coef_)
                    avgcoeff = np.mean(allcoefs,axis=0)
                    coefdict[cohort][sub][channel][features[featureidx]] = avgcoeff
                    performancedict[cohort][sub][channel]['ba'][features[featureidx]] = np.mean(scores)
                    performancedict[cohort][sub][channel]['95%CI'][features[featureidx]] = np.std(scores)*2
                performancedict[cohort][sub][channel]['explength'] = len(y_concat)
                performancedict[cohort][sub][channel]['movsamples'] = np.sum(y_concat)

np.save(r'D:\Glenn\AllfeaturesPerformances_l1.npy', performancedict)

np.save(r'D:\Glenn\modelcoeffs_l1.npy', coefdict)

# TODO: Leave out MedOn for subject 14 of Berlin, the run that does not have movement (due to left arm being used for rotation instead of right)
# TODO: Leave ALL OF sub EL015 --> Also no movement in label for MedOn and MedOff
# TODO: Leave out MedOff for sub EL016 --> No movement label