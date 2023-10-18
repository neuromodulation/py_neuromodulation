import numpy as np
import os
import csv
import sys

valtype = ['_leave_1_cohort_out','_leave_1_sub_out_across_coh','_leave_1_sub_out_within_coh']
# import the data
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", "2023_10_10-09_54"+valtype[0]+".npy"),
    allow_pickle="TRUE",
).item()

features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']

fields = ['cohort', 'sub', 'ch', 'type', 'ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined','95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined', 'length', 'movsamples']
fields = ['cohort', 'sub', 'ch', 'type', 'ba_combined', 'importance', '95%CI_combined', 'length', 'movsamples']
fields = ['cohort', 'sub', 'ba_combined']
f = open(r'C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAGRUcoh.csv', 'w',newline="")
w = csv.writer(f)

w.writerow(fields)

for cohort in ch_all.keys():
    if isinstance(ch_all[cohort],dict):
        for sub in ch_all[cohort].keys():
            if isinstance(ch_all[cohort][sub],dict):
                rowlist = []
                rowlist.append(cohort)
                rowlist.append(sub)
                print(ch_all[cohort][sub]['performance'])
                rowlist.append(ch_all[cohort][sub]['performance'])
                # Write the channel
                w.writerow(rowlist)

f.close()