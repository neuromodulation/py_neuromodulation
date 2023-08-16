import numpy as np
import os
import pandas as pd
from mni_to_atlas import AtlasBrowser

ch_all = np.load(
    os.path.join(r"D:\Glenn", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
df_info = pd.read_csv(r"D:\Glenn\df_ch_performances.csv")

### Adapted script that takes undefined regions and finds the closest (euclidean) brain region
atlas = AtlasBrowser("AAL")
regions = atlas.find_regions(np.array([df_info['x'],df_info['y'],df_info['z']]).T,plot=False) # Coordinates is n x 3
atlas = AtlasBrowser("AAL3")
regions_3 = atlas.find_regions(np.array([df_info['x'],df_info['y'],df_info['z']]).T,plot=False) # Coordinates is n x 3

# Probably useful to at least assign a number to each brain region in the data --> Could be used as auxillary (and allow unique colors in the

df_info['AAL_Region'] = regions
df_info['AAL3_Region'] = regions_3


# Also add info without L-R information
# Check if all regions end with L or R
if all([x[-2:] in ["_L","_R"] for x in regions]):
    newregions = [x[:-2] for x in regions]
    newregions_3 = [x[:-2] for x in regions_3]

df_info['AAL_absRegion'] = newregions
df_info['AAL3_absRegion'] = newregions_3

df_info.to_csv("df_ch_performances_regions.csv")