import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

ch_all = np.load(
    os.path.join(r"D:\Glenn", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
df_info = pd.read_csv(r"D:\Glenn\df_ch_performances_regions.csv")

plotting = True

# Plot the distribution of brain regions where electrodes are for the dataset
if plotting:
    #df_info['AAL_absRegion'].value_counts().plot(kind='bar')
    #df_info['AAL3_absRegion'].value_counts().plot(kind='bar')
    ch_persub = df_info.groupby('sub')['ch'].count().plot(kind='bar')
    plt.show()


# Find brain regions that >x% of the subjects have an electrode on, and find distribution of # of electrodes in these regions over the subjects
nrsubs = len(df_info["sub"].unique())
nrsubspercohort = df_info.groupby('cohort')['sub'].nunique()
uniqueregions = df_info["AAL_absRegion"].unique()
uniquecohorts = df_info['cohort'].unique()

subsperarea = df_info.groupby('AAL_absRegion')['sub'].nunique()
subspacohort = df_info.groupby(['AAL_absRegion','cohort'])['sub'].nunique()
# Now finds # of subjects that each area is part of --> Change to also reflect the cohort (To check whether we are not going to throw away a cohort)

subsperarea.divide(nrsubs).sort_values(ascending=False).plot(kind='bar')

dict = {}
for cohortname in uniquecohorts:
    cohortdata = []
    for region in uniqueregions:
        try:
            cohortdata.append(subspacohort[region][cohortname] / nrsubspercohort[cohortname]) # percentage of the cohort that has this region
            print(subspacohort[region][cohortname])
        except:
            cohortdata.append(0)
    dict[cohortname] = cohortdata
df = pd.DataFrame(dict,index=uniqueregions)
df.sort_values(by='Washington',ascending=False).plot(kind='bar')

