import numpy as np
import os
import csv
import sys

# import the data
ch_full = np.load(
    os.path.join(r"D:\Glenn", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

ch_all = np.load(
    os.path.join(r"D:\Glenn", "modelcoeffs_l1_2.npy"),
    allow_pickle="TRUE",
).item()

features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']
featuredim = ch_full['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[3] = np.add(idxlist_Berlin_001[3],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))
f = open(r'D:\Glenn\coeff_l1_2.csv', 'w',newline="")
w = csv.writer(f)

fields = ['cohort', 'sub', 'ch', 'type']

corrfeaturedim = [featuredim[i] for i in idxlist[-1]]
allfields = fields+ corrfeaturedim
w.writerow(allfields)

for cohort in ch_all.keys():
    for sub in ch_all[cohort].keys():
        for channel in ch_all[cohort][sub].keys():
            rowlist = []
            rowlist.append(cohort)
            rowlist.append(sub)
            rowlist.append(channel)
            if np.char.find(channel, 'ECOG') != -1:
                rowlist.append('ECOG')
            else:
                rowlist.append('LFP')
            try:
                for metric in [list(ch_all[cohort][sub][channel].keys())[-1]]:
                    for features in ch_all[cohort][sub][channel][metric][0]:
                        rowlist.append(features)
            except:
                continue
            else:
                # Write the channel
                w.writerow(rowlist)

f.close()