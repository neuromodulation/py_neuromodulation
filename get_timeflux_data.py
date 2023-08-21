import pandas as pd

fname = 'sub-749_ses-EcogLfpMedOff01_task-RealtimeDecodingR_acq-StimOff_run-1_ieeg.hdf5'

store = pd.HDFStore(fname, "r")
hdf_keys = list(store.keys())

# ['/features', '/prediction', '/rawdata']


store.close()

"python -m timeflux.helpers.viz timeflux_decoding.yaml"

