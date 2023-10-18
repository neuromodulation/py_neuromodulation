import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import xgboost
from sklearn.utils import class_weight
from sklearn.feature_selection import mutual_info_regression
from Experiments.utils.knn_bpp import kNN_BPP
# from einops.layers.torch import Rearrange # --> Can add to the model to reshape to fit 2dConv maybe
from matplotlib.colors import LogNorm

ch_all_train = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

ch_all_test = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "test_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

ch_all_feat = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "TempCleaned2_channel_all.npy"), # TempCleaned2_channel_all.npy
    allow_pickle="TRUE",
).item()
df_best_rmap_new = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch_adapted.csv")
df_best_rmap = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch.csv")
df_updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")

features = ['Hjorth','fft', 'Sharpwave', 'fooof', 'bursts']
performances = []
featuredim = ch_all_feat['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
featuredict = {}
for i in range(len(features)):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    featuredict[features[i]] = idx_i
toselect = ['fft']
idxlist = []
for featsel in toselect:
    idxlist.append(featuredict[featsel])
idxlist = np.concatenate(idxlist)

cohorts = [ "Pittsburgh", "Berlin","Beijing", "Washington"]

unequalmovcount = 0
unequalmovinfo = []
secrowdiffsignfft = 0
secrowdiffsignfftinfo = []
difflen = 0
diffleninfo = []
MImattot = np.zeros((36,36))
for coh in ch_all.keys():
    for sub in ch_all[coh].keys():
        for ch in ch_all[coh][sub].keys():
            # Old data
            x_all_concat = []
            y_all_concat = []
            for runs in ch_all_feat[coh][sub][ch].keys():
                x_all_concat.append(ch_all_feat[coh][sub][ch][runs]["data"])
                y_all_concat.append(ch_all_feat[coh][sub][ch][runs]["label"])

            if len(x_all_concat) > 1:
                x_all_concat = np.concatenate(x_all_concat, axis=0)
                y_all_concat = np.concatenate(y_all_concat, axis=0)

            else:
                x_all_concat = x_all_concat[0]
                y_all_concat = y_all_concat[0]
            # New data
            x_fft_concat = []
            y_fft_concat = []
            for runs in ch_all[coh][sub][ch].keys():
                x_fft_concat.append(ch_all[coh][sub][ch][runs]["data"])
                y_fft_concat.append(ch_all[coh][sub][ch][runs]["label"])

            if len(x_all_concat) > 1:
                x_fft_concat = np.concatenate(x_fft_concat, axis=0)
                y_fft_concat = np.concatenate(y_fft_concat, axis=0)

            else:
                x_fft_concat = x_fft_concat[0]
                y_fft_concat = y_fft_concat[0]
            print(coh+sub+ch)
            #MI between features
            if sub != '000':
                sub_num = sub.lstrip('0')
            else:
                sub_num = '0'
            if df_best_rmap_new.query("cohort == @coh and sub == @sub_num")["ch"].iloc[0] == ch:
                MImatcurr = np.zeros((36,36))
                for i in range(36):
                    MImatcurr[i,i:] = mutual_info_regression(x_all_concat[:,i:],x_all_concat[:,i])
                MImatcurr = MImatcurr + MImatcurr.T - np.diag(np.diag(MImatcurr))
                # Scale such that self-MI is 1 ?
                MInorm = MImatcurr / np.max(MImatcurr)
                MImattot += MInorm
                if False:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    cax = ax.matshow(MImatcurr, norm=LogNorm(vmin=np.min(MImatcurr) + 0.1, vmax=np.max(MImatcurr)))
                    ax.set_xticks(range(36))
                    ax.set_yticks(range(36))
                    ax.set_xticklabels(featuredim,rotation=45,ha='left')
                    ax.set_yticklabels(featuredim)
                    fig.colorbar(cax)
                    #plt.show()
                # Do some analysis
                if False:
                    if len(y_fft_concat) == len(y_all_concat):
                        a_yequal = y_fft_concat == y_all_concat
                        a_ycheck = len(y_fft_concat) == sum(a_yequal)
                        x_all_fftized = x_all_concat[:,idxlist]
                        a_xequal = x_fft_concat == x_all_fftized
                        if not a_ycheck:
                            unequalmovcount += 1
                            unequalmovinfo.append(coh+sub+ch)
                        fig, axs = plt.subplots(2, 7)

                        for i in range(7):
                            axs[0, i].plot(x_fft_concat[1000:2000, i])
                            rho = np.corrcoef(x_fft_concat[:,i],x_all_fftized[:,i])
                            axs[0, i].title.set_text(f'corrcoef = {rho[1,0]:.2f}')
                        for i in range(7):
                            axs[1, i].plot(x_all_fftized[1000:2000, i])
                        #plt.show()
                    else:
                        difflen += 1
                        diffleninfo.append(coh+sub+ch)
                    if x_fft_concat[1,0] < 0:
                        if not all(x_fft_concat[1,:] < 0):
                            secrowdiffsignfft += 1
                            secrowdiffsignfftinfo.append(coh+sub+ch)
                    else:
                        if not all(x_fft_concat[1, :] > 0):
                            secrowdiffsignfft += 1
                            secrowdiffsignfftinfo.append(coh + sub + ch)
