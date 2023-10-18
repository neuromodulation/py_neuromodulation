# Can try to assign different levels of parcellation for creating brain regions --> Could be useful to create some diversity in motor cortex
# Maybe 128 or 256 Difumo (Dictionary of functional modes) parcellation --> Atlas to extract functional signals
# Optimized to represent fMRI signals

# Or would one of the Julich-Brains work (Cytoarchitectonically defined regions)
import siibra
import numpy as np
import pandas as pd


df_info = pd.read_csv(r"D:\Glenn\df_ch_performances_regions.csv")

difumo = siibra.get_map(
    parcellation="DIFUMO_ATLAS_128_DIMENSIONS", #Choose from 64, 128, 256 etc. (would not think higher is usefull in this case)
    space="mni152",
    maptype="labelled"
)

pointlist = [tuple(i) for i in np.array([df_info['x'],df_info['y'],df_info['z']]).T]

point = siibra.PointSet(pointlist, space='mni152')
assignments = difumo.assign(point)

# Fill in some channels by supplying an inaccuracy in its position and taking the highest correlating region
for input in range(df_info.shape[0]):
    if input not in list(assignments['input structure']):
        # Find the row number of the previous point IN the dataset
        try:
            rownr = int(assignments[assignments['input structure'] == input-1].index[0])
        except:
            rownr = 0
        # add value to sigma until some areas are found
        sigma = 6
        while True:
            try:
                point = siibra.Point(pointlist[input], space='mni152',sigma_mm=sigma)
                assign = difumo.assign(point)
                assign = assign.sort_values(by='correlation',ascending=False).iloc[[0]]
                assign['input structure'] = input
                assignments = pd.concat([assignments.iloc[:rownr+1], assign, assignments.iloc[rownr+1:]]).reset_index(drop=True)
            except:
                sigma = sigma+2
                continue
            else:
                # When no exception occured break the loop
                break

## Evaluate the assignment
# Number of subs per region
assignments['region'].value_counts().plot(kind='bar')
# Concatenate right / left parts
newname = []
for i in assignments['region']:
    name = i.name
    words = name.split()
    if words[-1] in ['left','right']:
        newname.append(' '.join(words[0:-1]))
    else:
        newname.append(name)