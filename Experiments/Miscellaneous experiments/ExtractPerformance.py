import numpy as np
import os

cohdat = np.load(os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances",'2023_09_14-15_36_leave_1_cohort_out.npy'),allow_pickle="TRUE").item()
acrossdat = np.load(os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances",'2023_09_14-15_36_leave_1_sub_out_across_coh.npy'),allow_pickle="TRUE").item()


