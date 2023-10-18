from sklearn.model_selection import KFold, cross_validate, cross_val_score
import numpy as np
import os
import pandas as pd
from torch import nn
import torch
import cebra
import cebra.models
import cebra.data
import cebra.distributions
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin, cebra_layers
from sklearn import metrics, neighbors
from sklearn import linear_model
from Experiments.utils.cebracustom import CohortDiscreteDataLoader
from Experiments.utils.ExtraTorchFunc import Attention, WeightAttention, MatrixAttention, AttentionWithContext
torch.backends.cudnn.benchmark = True

# import the data
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "TempCleaned_channel_all.npy"),
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
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
performancedict = {}
meanlist = []
features = ['combined']
for cohort in ch_all.keys():
    print(cohort)
    performancedict[cohort] = {}
    for sub in ch_all[cohort].keys():
        print(sub)
        performancedict[cohort][sub] = {}
        for chname in ch_all[cohort][sub].keys():
            performancedict[cohort][sub][chname] = {}
            performancedict[cohort][sub][chname]['ba'] = {}
            performancedict[cohort][sub][chname]['95%CI'] = {}
            x_concat = []
            y_concat = []
            for runs in ch_all[cohort][sub][chname].keys():
                x_concat.append(
                    np.squeeze(ch_all[cohort][sub][chname][runs]['data']))  # Get best ECOG data
                y_concat.append(ch_all[cohort][sub][chname][runs]['label'])
            x_concat = np.concatenate(x_concat, axis=0)
            y_concat = np.concatenate(y_concat, axis=0)
            crossvals = kf.split(x_concat,y_concat)
            scores = cross_val_score(model, x_concat, np.array(y_concat, dtype=int), cv=kf, scoring=bascorer)
            performancedict[cohort][sub][chname]['ba'][features[0]] = np.mean(scores)
            #print(np.mean(scores))
            meanlist.append(np.mean(scores))
            #print(f'Running mean: {np.mean(meanlist)}')
            performancedict[cohort][sub][chname]['95%CI'][features[0]] = np.std(scores) * 2
            performancedict[cohort][sub][chname]['explength'] = len(y_concat)
            performancedict[cohort][sub][chname]['movsamples'] = np.sum(y_concat)

np.save(r'C:\Users\ICN_GPU\Documents\Glenn_Data\AllfeaturePerformances_TempCleaned.npy', performancedict)
