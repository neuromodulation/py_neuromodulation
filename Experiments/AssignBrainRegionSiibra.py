# Can try to assign different levels of parcellation for creating brain regions --> Could be useful to create some diversity in motor cortex
# Maybe 128 or 256 Difumo (Dictionary of functional modes) parcellation --> Atlas to extract functional signals
# Optimized to represent fMRI signals

# Or would one of the Julich-Brains work (Cytoarchitectonically defined regions)
import siibra
from nilearn import plotting


difumo = siibra.get_map(
    parcellation="DIFUMO_ATLAS_128_DIMENSIONS", #Choose from 64, 128, 256 etc. (would not think higher is usefull in this case)
    space="mni152",
    maptype="labelled"
)

point = siibra.Point((27.75, -32.0, 63.725), space='mni152')
with siibra.QUIET:  # suppress progress output
    assignments = difumo.assign(point)
