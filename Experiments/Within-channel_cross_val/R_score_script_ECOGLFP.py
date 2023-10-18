from sklearn import metrics, model_selection, linear_model, svm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import xgboost
from sklearn.utils import class_weight

# import the data
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

df_perf = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")


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
model = linear_model.LogisticRegression(class_weight="balanced",max_iter=1000)
#model = svm.SVC(class_weight="balanced", C=0.5, cache_size = 5000) # RBF one performs best (vs Log) with C=0.25 but slow
#model = svm.LinearSVC(class_weight="balanced",dual='auto', C=0.1)
#RBFapprox = kernel_approximation.RBFSampler(gamma='scale', n_components=800) # Can then use linearSVC, which performs worse than SVC(linear)
#model = ensemble.BaggingClassifier(estimator=model,n_estimators=5) # Does not seem to speedup SVC
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
# loop over all bestECOG+'+'+bestLFPs
performancedict = {}

coef = False
XGB = False
coefdict = {} # Will just be empty otherwise
features = ['combined']
for cohort in ['Berlin','Beijing','Pittsburgh']:
    print(cohort)
    performancedict[cohort] = {}
    coefdict[cohort] = {}
    for sub in ch_all[cohort].keys():
        print(sub)
        performancedict[cohort][sub] = {}
        coefdict[cohort][sub] = {}

        if cohort == 'Berlin' and sub == '001':
            bestECOG = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'ECOG'").sort_values(
                    by='ba_combined', ascending=False)[
                    "ch"].iloc[0]
            LFPranked = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'LFP'").sort_values(
                by='ba_combined', ascending=False)[
                "ch"]
            # Create set of the runs contained
            ECOGrunset = set(ch_all[cohort][sub][bestECOG].keys())

            # Select those bestECOG+'+'+bestLFPs that LFP also had
            for i in range(len(LFPranked)):
                bestLFP = LFPranked.iloc[i]
                LFPruns = np.array(list(ch_all[cohort][sub][bestLFP].keys()))
                contains = [ele in ECOGrunset for ele in LFPruns] # list of True/False, for each ch in ECOG whether it exists in LFP
                if np.sum(contains) == 0:
                    continue
                else: # Then ECOG has at least one bestECOG+'+'+bestLFP that LFP also had
                    overlapruns = LFPruns[contains]
                    break
            # Run estimator --> Maybe with a feature selector protocol?
            performancedict[cohort][sub][bestECOG+'+'+bestLFP] = {}
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['ba'] = {}
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['95%CI'] = {}
            coefdict[cohort][sub][bestECOG+'+'+bestLFP] = {}
            for featureidx in range(len(features)):
                x_concat = []
                y_concat = []
                for runs in overlapruns:
                    x_temp = np.concatenate(
                        (np.squeeze(ch_all[cohort][sub][bestECOG][runs]['data'][:,idxlist_Berlin_001[5]]),
                         np.squeeze(ch_all[cohort][sub][bestLFP][runs]['data'][:, idxlist_Berlin_001[5]])),axis=1)
                    x_concat.append(x_temp) # Get best ECOG data
                    y_concat.append(ch_all[cohort][sub][bestECOG][runs]['label'])
                x_concat = np.concatenate(x_concat, axis=0)
                y_concat = np.concatenate(y_concat, axis=0)
                if coef:
                    cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                    scores = cv_out['test_score']
                    allcoefs = []
                    for model in cv_out['estimator']:
                        allcoefs.append(model.coef_)
                    avgcoeff = np.mean(allcoefs,axis=0)
                    coefdict[cohort][sub][bestECOG+'+'+bestLFP][features[featureidx]] = avgcoeff
                else:
                    # decoder.set_params(**{'lambda':2})
                    #classes_weights = class_weight.compute_sample_weight(
                    #    class_weight="balanced", y=y_concat
                    #)
                    #ratio = np.sum(y_concat==0)/np.sum(y_concat==1)
                    #decoder = xgboost.XGBClassifier()
                    #decoder.set_params(eval_metric="logloss", scale_pos_weight=ratio)

                    scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer)
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['ba'][features[featureidx]] = np.mean(scores)
                print(np.mean(scores))
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['95%CI'][features[featureidx]] = np.std(scores)*2
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['explength'] = len(y_concat)
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['movsamples'] = np.sum(y_concat)
        elif cohort == 'Berlin' and sub == '014':
            bestLFP = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'LFP'").sort_values(
                    by='ba_combined', ascending=False)[
                    "ch"].iloc[0]
            ECOGranked = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'ECOG'").sort_values(
                by='ba_combined', ascending=False)[
                "ch"]
            # Create set of the runs contained
            LFPrunset = set([list(ch_all[cohort][sub][bestLFP].keys())[0]])

            # Select those bestECOG+'+'+bestLFPs that LFP also had
            for i in range(len(ECOGranked)):
                bestECOG = ECOGranked.iloc[i]
                ECOGruns = np.array([list(ch_all[cohort][sub][bestECOG].keys())[0]])
                contains = [ele in LFPrunset for ele in ECOGruns] # list of True/False, for each ch in ECOG whether it exists in LFP
                if np.sum(contains) == 0:
                    continue
                else: # Then ECOG has at least one bestECOG+'+'+bestLFP that LFP also had
                    overlapruns = ECOGruns[contains]
                    break
            performancedict[cohort][sub][bestECOG + '+' + bestLFP] = {}
            performancedict[cohort][sub][bestECOG + '+' + bestLFP]['ba'] = {}
            performancedict[cohort][sub][bestECOG + '+' + bestLFP]['95%CI'] = {}
            coefdict[cohort][sub][bestECOG + '+' + bestLFP] = {}
            for featureidx in range(len(features)):
                x_concat = []
                y_concat = []
                for runs in overlapruns:
                    x_temp = np.concatenate(
                        (np.squeeze(ch_all[cohort][sub][bestECOG][runs]['data'][:,idxlist[5]]),
                         np.squeeze(ch_all[cohort][sub][bestLFP][runs]['data'][:, idxlist[5]])),axis=1)
                    x_concat.append(x_temp) # Get best ECOG data
                    y_concat.append(ch_all[cohort][sub][bestECOG][runs]['label'])
                x_concat = np.concatenate(x_concat, axis=0)
                y_concat = np.concatenate(y_concat, axis=0)
                if coef:
                    cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                    scores = cv_out['test_score']
                    allcoefs = []
                    for model in cv_out['estimator']:
                        allcoefs.append(model.coef_)
                    avgcoeff = np.mean(allcoefs,axis=0)
                    coefdict[cohort][sub][bestECOG+'+'+bestLFP][features[featureidx]] = avgcoeff
                else:
                    scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf,
                                             scoring=bascorer)
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['ba'][features[featureidx]] = np.mean(scores)
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['95%CI'][features[featureidx]] = np.std(scores)*2
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['explength'] = len(y_concat)
            performancedict[cohort][sub][bestECOG+'+'+bestLFP]['movsamples'] = np.sum(y_concat)
        elif cohort == 'Berlin' and sub == 'EL015':
            del coefdict[cohort][sub]
            del performancedict[cohort][sub]
        elif cohort == 'Berlin' and sub == 'EL016':
            LFPranked = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'LFP'").sort_values(
                by='ba_combined', ascending=False)[
                "ch"]
            ECOGranked = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'ECOG'").sort_values(
                by='ba_combined', ascending=False)[
                "ch"]
            # Create set of the runs contained
            for i in range(len(LFPranked)):
                bestLFP = LFPranked.iloc[i]
                LFPrunlist = list(ch_all[cohort][sub][bestLFP].keys())
                medoffonly = [run for run in LFPrunlist if (np.char.find(run, 'MedOn') == -1)]
                if medoffonly:
                    LFPrunset = set(medoffonly)
                    break
                else:
                    continue
            # Select those bestECOG+'+'+bestLFPs that LFP also had
            for i in range(len(ECOGranked)):
                bestECOG = ECOGranked.iloc[i]
                ECOGruns = np.array(list(ch_all[cohort][sub][bestECOG].keys()))
                contains = [ele in LFPrunset for ele in
                            ECOGruns]  # list of True/False, for each ch in ECOG whether it exists in LFP
                if np.sum(contains) == 0:
                    continue
                else:  # Then ECOG has at least one bestECOG+'+'+bestLFP that LFP also had
                    overlapruns = ECOGruns[contains]
                    break
            performancedict[cohort][sub][bestECOG + '+' + bestLFP] = {}
            performancedict[cohort][sub][bestECOG + '+' + bestLFP]['ba'] = {}
            performancedict[cohort][sub][bestECOG + '+' + bestLFP]['95%CI'] = {}
            coefdict[cohort][sub][bestECOG + '+' + bestLFP] = {}
            try: # Just to get around the one channel that does not have MedOn
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in overlapruns: # Only include med on (which should be first in the keylist)
                        if np.char.find(runs, 'MedOn') != -1:
                            x_temp = np.concatenate(
                                (np.squeeze(
                                    ch_all[cohort][sub][bestECOG][runs]['data'][:, idxlist[5]]),
                                 np.squeeze(
                                     ch_all[cohort][sub][bestLFP][runs]['data'][:, idxlist[5]])),
                                axis=1)
                            x_concat.append(x_temp)  # Get best ECOG data
                            y_concat.append(ch_all[cohort][sub][bestECOG][runs]['label'])
                        else:
                            continue
                    x_concat = np.concatenate(x_concat, axis=0)
                    y_concat = np.concatenate(y_concat, axis=0)
                    if coef:
                        cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                        scores = cv_out['test_score']
                        allcoefs = []
                        for model in cv_out['estimator']:
                            allcoefs.append(model.coef_)
                        avgcoeff = np.mean(allcoefs, axis=0)
                        coefdict[cohort][sub][bestECOG+'+'+bestLFP][features[featureidx]] = avgcoeff
                    else:
                        scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf,
                                                 scoring=bascorer)
                    performancedict[cohort][sub][bestECOG+'+'+bestLFP]['ba'][features[featureidx]] = np.mean(scores)
                    performancedict[cohort][sub][bestECOG+'+'+bestLFP]['95%CI'][features[featureidx]] = np.std(scores)*2
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['explength'] = len(y_concat)
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['movsamples'] = np.sum(y_concat)
            except:
                del performancedict[cohort][sub][bestECOG+'+'+bestLFP]
        else:
            try:
                bestECOG = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'ECOG'").sort_values(
                    by='ba_combined', ascending=False)[
                    "ch"].iloc[0]
                LFPranked = df_perf.query(f"cohort == @cohort and sub == @sub and type == 'LFP'").sort_values(
                    by='ba_combined', ascending=False)[
                    "ch"]
                # Create set of the runs contained
                ECOGrunset = set(ch_all[cohort][sub][bestECOG].keys())
                # Select those bestECOG+'+'+bestLFPs that LFP also had
                for i in range(len(LFPranked)):
                    bestLFP = LFPranked.iloc[i]
                    LFPruns = np.array(list(ch_all[cohort][sub][bestLFP].keys()))
                    contains = [ele in ECOGrunset for ele in
                                LFPruns]  # list of True/False, for each ch in ECOG whether it exists in LFP
                    if np.sum(contains) == 0:
                        continue
                    else:  # Then ECOG has at least one bestECOG+'+'+bestLFP that LFP also had
                        overlapruns = LFPruns[contains]
                        break
                performancedict[cohort][sub][bestECOG + '+' + bestLFP] = {}
                performancedict[cohort][sub][bestECOG + '+' + bestLFP]['ba'] = {}
                performancedict[cohort][sub][bestECOG + '+' + bestLFP]['95%CI'] = {}
                coefdict[cohort][sub][bestECOG + '+' + bestLFP] = {}
                for featureidx in range(len(features)):
                    x_concat = []
                    y_concat = []
                    for runs in overlapruns:
                        x_temp = np.concatenate(
                            (np.squeeze(ch_all[cohort][sub][bestECOG][runs]['data'][:, idxlist[5]]),
                             np.squeeze(ch_all[cohort][sub][bestLFP][runs]['data'][:, idxlist[5]])), axis=1)
                        x_concat.append(x_temp)  # Get best ECOG data
                        y_concat.append(ch_all[cohort][sub][bestECOG][runs]['label'])
                    x_concat = np.concatenate(x_concat,axis=0)
                    y_concat = np.concatenate(y_concat,axis=0)
                    if coef:
                        cv_out = cross_validate(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring = bascorer,return_estimator=True)
                        scores = cv_out['test_score']
                        allcoefs = []
                        for model in cv_out['estimator']:
                            allcoefs.append(model.coef_)
                        avgcoeff = np.mean(allcoefs,axis=0)
                        coefdict[cohort][sub][bestECOG+'+'+bestLFP][features[featureidx]] = avgcoeff
                    else:
                        scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf,
                                                 scoring=bascorer)
                    print(np.mean(scores))
                    performancedict[cohort][sub][bestECOG+'+'+bestLFP]['ba'][features[featureidx]] = np.mean(scores)
                    performancedict[cohort][sub][bestECOG+'+'+bestLFP]['95%CI'][features[featureidx]] = np.std(scores)*2
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['explength'] = len(y_concat)
                performancedict[cohort][sub][bestECOG+'+'+bestLFP]['movsamples'] = np.sum(y_concat)
            except:
                del coefdict[cohort][sub]
                del performancedict[cohort][sub]

np.save(r'C:\Users\ICN_GPU\Documents\Glenn_Data\LFPECOGperformance.npy', performancedict)
if coef:
    np.save(r'C:\Users\ICN_GPU\Documents\Glenn_Data\modelcoeffs_l1_2.npy', coefdict)

# TODO: Leave out MedOn for subject 14 of Berlin, the run that does not have movement (due to left arm being used for rotation instead of right)
# TODO: Leave ALL OF sub EL015 --> Also no movement in label for MedOn and MedOff
# TODO: Leave out MedOff for sub EL016 --> No movement label