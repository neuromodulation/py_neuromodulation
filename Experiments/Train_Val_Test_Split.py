# File/function to generate (consistent) random train_val_test split with proper distribution over different cohorts,
# retaining similar subject per cohort balance in all sets. --> Do as K-Fold (i.e. fully separate test set, and then
# rotate through the training set (splitting a part in test)

# Thus: need consistent seperated test indices and then in the run_cross_val do K-Fold