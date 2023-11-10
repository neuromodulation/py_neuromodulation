import numpy as np
import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

timestamp = '2023_11_07-10_30_'
allvaltypes = ["leave_1_sub_out_across_coh","leave_1_sub_out_within_coh","leave_1_cohort_out"]
curtypes = [1]
curvaltype = np.array(allvaltypes)[curtypes]
plot = False # Adapt title and such

if 0 in curtypes:
    dat = np.load(os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances",
                                     timestamp + 'leave_1_sub_out_across_coh.npy'), allow_pickle="TRUE").item()
if 1 in curtypes:
    dat = np.load(os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances",timestamp+'leave_1_sub_out_within_coh.npy'),allow_pickle="TRUE").item()
if 2 in curtypes:
    dat = np.load(os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances",timestamp+'leave_1_cohort_out.npy'),allow_pickle="TRUE").item()

if plot:
    Col = ['Cohort','Validation','Performance']
    df = pd.DataFrame(columns=['Cohort','Validation','Performance'])
    for cohort in ['Beijing','Pittsburgh','Berlin','Washington']:
        dataset = dat
        result = [[cohort,np.array(allvaltypes)[curtypes][0],dataset[cohort][item]['performance']] for item in dataset[cohort] if isinstance(dataset[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result,columns=Col)
        df = pd.concat([df,df_temp])

    plt.figure()
    g = sns.boxplot(df,y='Performance',x='Cohort',palette='viridis',order=['Pittsburgh', 'Berlin', 'Beijing', 'Washington'])
    g.set_title('Generalized, across-cohort decoding',fontsize=17)
    g.set(ylim=(0.49, 1))
    g.axhline(0.5,ls='--')
    g.set_xlabel("Cohort",fontsize=17)
    g.set_ylabel("Balanced accuracy",fontsize=17)
    g.tick_params(labelsize=15)
    plt.savefig(r"C:\Users\ICN_GPU\Documents\Glenn_Data\Presentation\PerfAcrossContra_RTBayopt.svg")
    plt.show()