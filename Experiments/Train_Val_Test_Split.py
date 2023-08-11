# File/function to generate (consistent) random train_val_test split with proper distribution over different cohorts,
# retaining similar subject per cohort balance in all sets. --> Do as K-Fold (i.e. fully separate test set, and then
# rotate through the training set (splitting a part in test)

# Thus: need consistent seperated test indices and then in the run_cross_val do K-Fold
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
import copy

ch_all = np.load(
    os.path.join(r"D:\Glenn", "channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

cohorts = ["Beijing", "Pittsburgh", "Berlin", ]  # "Washington"

nrsubspercohort = []
pretend_data = []
cohort_label = []
cohortlist = []
keylist = []
cohortnr = 1
for cohort in cohorts:
    nrsubs = 0
    for subs in ch_all[cohort].keys():
        nrsubs += 1
        pretend_data.append([1])
        cohort_label.append(cohortnr)
        cohortlist.append(cohort)
        keylist.append(subs)
    cohortnr += 1
    nrsubspercohort.append(nrsubs)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
print(sss.split(pretend_data, cohort_label))


for i, (train_index, test_index) in enumerate(sss.split(pretend_data, cohort_label)):
    test_idx = test_index
    train_idx = train_index

test_dict = copy.deepcopy(ch_all)
train_dict = copy.deepcopy(ch_all)
for idx in train_idx:
    del test_dict[cohortlist[idx]][keylist[idx]]

for idx in test_idx:
    del train_dict[cohortlist[idx]][keylist[idx]]

save = True
if save:
   np.save(os.path.join(r"D:\Glenn", "train_channel_all_fft.npy"), train_dict, allow_pickle="TRUE")
   np.save(os.path.join(r"D:\Glenn", "test_channel_all_fft.npy"), test_dict, allow_pickle="TRUE")
