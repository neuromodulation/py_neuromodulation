import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_info = pd.read_csv(r"D:\Glenn\df_ch_performances_regions_4.csv")

plotting = True
removeLR = True

atlas = 'DiFuMo256'
# Plot the distribution of brain regions where electrodes are for the dataset
if removeLR:
    newname = []
    for i in df_info[atlas]:
        words = i.split()
        if words[-1] in ['left','right']:
            newname.append(' '.join(words[0:-1]))
        else:
            newname.append(i)
    df_info[atlas] = newname
if plotting:
    df_info[atlas].value_counts().plot(kind='bar')
    #ch_persub = df_info.groupby('sub')['ch'].count().plot(kind='bar')
    plt.show()


# Find brain regions that >x% of the subjects have an electrode on, and find distribution of # of electrodes in these regions over the subjects
nrsubs = len(df_info["sub"].unique())
nrsubspercohort = df_info.groupby('cohort')['sub'].nunique()
uniqueregions = df_info[atlas].unique()
uniquecohorts = df_info['cohort'].unique()

subsperarea = df_info.groupby(atlas)['sub'].nunique()
subspacohort = df_info.groupby([atlas,'cohort'])['sub'].nunique()
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
# Drop regions that eliminate 1 of the main cohorts
df.drop(df[np.logical_or(np.logical_or(list(df['Berlin'] < 0.01), list(df['Beijing']<0.01)),list(df['Pittsburgh']<0.01))].index, inplace = True)
df.sort_values(by='Berlin',ascending=False).plot(kind='bar', rot=45)
plt.xticks(ha='right')