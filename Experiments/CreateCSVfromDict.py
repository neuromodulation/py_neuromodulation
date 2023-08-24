import numpy as np
import os
import csv
import sys

# import the data
ch_all = np.load(
    os.path.join(r"D:\Glenn", "AllfeaturesPerformances_correctlength.npy"),
    allow_pickle="TRUE",
).item()

features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']

fields = ['cohort', 'sub', 'ch', 'type', 'ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined','95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined', 'length', 'movsamples']

f = open(r'D:\Glenn\df_all_features.csv', 'w',newline="")
w = csv.writer(f)

w.writerow(fields)

for cohort in ch_all.keys():
    for sub in ch_all[cohort].keys():
        for channel in ch_all[cohort][sub].keys():
            if np.char.find(channel, 'explength') == -1 and np.char.find(channel, 'movsamples') == -1:
                rowlist = []
                rowlist.append(cohort)
                rowlist.append(sub)
                rowlist.append(channel)
                if np.char.find(channel, 'ECOG') != -1:
                    rowlist.append('ECOG')
                else:
                    rowlist.append('LFP')
                for metric in list(ch_all[cohort][sub][channel].keys())[:-2]:
                    for features in ch_all[cohort][sub][channel][metric].keys():
                        rowlist.append(ch_all[cohort][sub][channel][metric][features])
                # Write the channel
                rowlist.append(ch_all[cohort][sub][channel]['explength'])
                rowlist.append(ch_all[cohort][sub][channel]['movsamples'])
                w.writerow(rowlist)

f.close()