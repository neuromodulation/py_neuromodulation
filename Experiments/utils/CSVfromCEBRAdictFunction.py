import numpy as np
import os
import csv
import sys

def CSVfromCEBRAdictFunction(valtypes,timestring):
    '''Writes a CSV with the CEBRA performances
    Input: List of valtypes of the run
    string contraining the timestamp of the run
    Output: CSV file in Glenn_Data/CSVs/timestamp_validationtype.csv'''

    # import the data
    for curtype in valtypes:
        ch_all = np.load(
            os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", timestring+'_'+curtype+".npy"),
            allow_pickle="TRUE",
        ).item()

        features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']

        fields = ['cohort', 'sub', 'ba_combined']
        f = open(r'C:\Users\ICN_GPU\Documents\Glenn_Data\ '+timestring+'_'+curtype+ '.csv', 'w',newline="")
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