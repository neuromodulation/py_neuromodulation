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
uniqueregions = df_info["AAL3_absRegion"].unique()

subsperarea = df_info.groupby('AAL3_absRegion')['sub'].value_counts()
# Now finds # of subjects that each area is part of --> Change to also reflect the cohort (To check whether we are not going to throw away a cohort)
dict = {}
for region in uniqueregions:
    dict[region] = subsperarea[region].size
df = pd.DataFrame.from_dict(dict,orient='index')
df.plot(kind='bar')
