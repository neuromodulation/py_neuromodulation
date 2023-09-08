import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
# Load the csv as a dataframe
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")

perflist = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined','95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined', 'length', 'movsamples']
baonly = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined']
baonly = ['ba_combined']
CIonly = ['95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined']
cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]

corr = df['ba_combined'].corr(df['length'])
# Corr per cohort
corrs = []
f, axes = plt.subplots(2, 2)
for i in range(len(cohorts)//2):
    for j in range(len(cohorts)//2):
        limdf = df[df["cohort"].str.contains(cohorts[i*2+j]) == True]
        corrs.append(limdf['ba_combined'].corr(limdf['length']))
        g = sns.regplot(x=limdf['length'],y=limdf['ba_combined'],ax=axes[i,j],robust=True)
        axes[i,j].set_title(cohorts[i*2+j])
        axes[i,j].set_xlabel('recording length')
        axes[i,j].set_ylabel('balanced accuracy')

        r, p = sp.stats.spearmanr(limdf['ba_combined'], limdf['length'])
        axes[i,j].text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                    transform=axes[i,j].transAxes)

