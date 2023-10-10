import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")

# Load the csv as a dataframe
perflist = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined','95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined', 'length', 'movsamples']
baonly = ['ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined']
baonly = ['ba_combined']
CIonly = ['95%CI_Hjorth', '95%CI_Sharpwave', '95%CI_fooof', '95%CI_bursts','95%CI_fft', '95%CI_combined']
cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]
# Best performing channel and corresponding feature per subject
meanperfpersub = df.groupby('sub')[perflist].mean()
meanperfpersubpertype = df.groupby(['type','sub'])[perflist].mean()
meanperfpercohort = df.groupby('cohort')[perflist].mean()
# mean of cohort means
meanofmeans = meanperfpercohort.mean()

# Plot and save tables

# Boxplot over best channels per subject
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")
idxofmax = np.sort(list(df.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersub = df.iloc[idxofmax][['cohort','ba_Hjorth', 'ba_Sharpwave', 'ba_fooof', 'ba_bursts','ba_fft', 'ba_combined']]
maxsingle = maxpersub.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
maxsingle['ba_combined'] = maxsingle['ba_combined'].replace('ba_combined','Within channel LogREG')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\WithinChannelCEBRA.csv")
baonly = ['ba_combined']
maxECOGLFP = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
maxECOGLFP['ba_combined'] = maxECOGLFP['ba_combined'].replace('ba_combined','Within channel CEBRA')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestWithin.csv")
CEBRAwithin = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAwithin['ba_combined'] = CEBRAwithin['ba_combined'].replace('ba_combined','CEBRA within cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestAcross.csv")
CEBRAacross = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAacross['ba_combined'] = CEBRAacross['ba_combined'].replace('ba_combined','CEBRA across cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestCoh.csv")
CEBRAcoh = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAcoh['ba_combined'] = CEBRAcoh['ba_combined'].replace('ba_combined','CEBRA leave cohort out')

comparisondf = pd.concat([maxsingle,maxECOGLFP,CEBRAwithin,CEBRAacross,CEBRAcoh])
g = sns.boxplot(data= comparisondf,x='ba_combined', y='balanced accuracy',hue='cohort')
g.set(ylim=(0.49, 1))
g.axhline(0.5,ls='--')

################## For the same subjects as the Test set #######################
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestCoh.csv")
cohlist = df['cohort']
sublist = df['sub']

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\AllfeaturePerformances_TempCleaned.csv")
idxofmax = np.sort(list(df.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersub = df.iloc[idxofmax][['cohort','sub', 'ba_combined']]
Testsel = pd.DataFrame()
for i in range(len(cohlist)):
    Testsel = pd.concat([Testsel, maxpersub[(df['cohort']==cohlist[i]) & (df['sub']==sublist[i])]], ignore_index=True)

maxsingle = Testsel.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
maxsingle['ba_combined'] = maxsingle['ba_combined'].replace('ba_combined','Within channel LogREG')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\WithinChannelCEBRA_RT.csv")
baonly = ['ba_combined']
Testsel = pd.DataFrame()
for i in range(len(cohlist)):
    Testsel = pd.concat([Testsel, df[(df['cohort']==cohlist[i]) & (df['sub']==sublist[i].lstrip('0'))]], ignore_index=True)
maxECOGLFP = Testsel.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
maxECOGLFP['ba_combined'] = maxECOGLFP['ba_combined'].replace('ba_combined','Within channel CEBRA')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestWithin.csv")
CEBRAwithin = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAwithin['ba_combined'] = CEBRAwithin['ba_combined'].replace('ba_combined','CEBRA within cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestAcross.csv")
CEBRAacross = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAacross['ba_combined'] = CEBRAacross['ba_combined'].replace('ba_combined','CEBRA across cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestCoh.csv")
CEBRAcoh = df.melt(id_vars='cohort',value_vars=baonly, var_name='ba_combined'
                                       , value_name='balanced accuracy')
CEBRAcoh['ba_combined'] = CEBRAcoh['ba_combined'].replace('ba_combined','CEBRA leave cohort out')

comparisondf = pd.concat([maxsingle,maxECOGLFP,CEBRAwithin,CEBRAacross,CEBRAcoh])
g = sns.boxplot(data= comparisondf,x='ba_combined', y='balanced accuracy',hue='cohort')
g.set(ylim=(0.49, 1))
g.axhline(0.5,ls='--')

### plot for the across cohort cebra the results over different cohorts
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestAcross.csv")
g = sns.boxplot(df,y='ba_combined',x='cohort',palette='viridis',order=['Pittsburgh', 'Berlin', 'Beijing', 'Washington'])
g.set_title('Generalized, across-cohort decoding',fontsize=17)
g.set(ylim=(0.49, 1))
g.axhline(0.5,ls='--')
g.set_xlabel("Cohort",fontsize=17)
g.set_ylabel("Balanced accuracy",fontsize=17)
g.tick_params(labelsize=15)
plt.savefig(r"C:\Users\ICN_GPU\Documents\Glenn_Data\Figures\PosterFigures\PerfAcrossContra_RT.pdf")

########################## Not per cohort ###########################
################## For the same subjects as the Test set #######################
df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestCoh.csv")
cohlist = df['cohort']
sublist = df['sub']

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\AllfeaturePerformances_TempCleaned.csv")
idxofmax = np.sort(list(df.groupby(['sub', 'cohort'])['ba_combined'].idxmax()))
maxpersub = df.iloc[idxofmax][['cohort','sub', 'ba_combined']]
Testsel = pd.DataFrame()
for i in range(len(cohlist)):
    Testsel = pd.concat([Testsel, maxpersub[(df['cohort']==cohlist[i]) & (df['sub']==sublist[i])]], ignore_index=True)

maxsingle = Testsel
maxsingle['ba_combined'] = maxsingle['ba_combined'].replace('ba_combined','Within channel LogREG')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\WithinChannelCEBRA_RT.csv")
baonly = ['ba_combined']
Testsel = pd.DataFrame()
for i in range(len(cohlist)):
    Testsel = pd.concat([Testsel, df[(df['cohort']==cohlist[i]) & (df['sub']==sublist[i].lstrip('0'))]], ignore_index=True)
maxECOGLFP = Testsel
maxECOGLFP['ba_combined'] = maxECOGLFP['ba_combined'].replace('ba_combined','Within channel CEBRA')

#df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\WithinChannelCEBRA_ECOGLFP.csv")
#baonly = ['ba_combined']
#Testsel = pd.DataFrame()
#for i in range(len(cohlist)):
#    Testsel = pd.concat([Testsel, df[(df['cohort']==cohlist[i]) & (df['sub']==sublist[i].lstrip('0'))]], ignore_index=True)
#WithinECOGLFPCEBRA = Testsel
#WithinECOGLFPCEBRA['ba_combined'] = WithinECOGLFPCEBRA['ba_combined'].replace('ba_combined','Within channel CEBRA')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestWithin.csv")
CEBRAwithin = df
CEBRAwithin['ba_combined'] = CEBRAwithin['ba_combined'].replace('ba_combined','CEBRA within cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestAcross.csv")
CEBRAacross = df
CEBRAacross['ba_combined'] = CEBRAacross['ba_combined'].replace('ba_combined','CEBRA across cohort')

df = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRAtestCoh.csv")
CEBRAcoh = df
CEBRAcoh['ba_combined'] = CEBRAcoh['ba_combined'].replace('ba_combined','CEBRA leave cohort out')
comparisondf = pd.DataFrame()
comparisondf['Within channel LogREG'] = maxsingle['ba_combined']
comparisondf['Within channel CEBRA'] = maxECOGLFP['ba_combined']
#comparisondf['Within channel CEBRA ECOG LFP'] = WithinECOGLFPCEBRA['ba_combined']
comparisondf['CEBRA within cohort'] = CEBRAwithin['ba_combined']
comparisondf['CEBRA across cohort'] = CEBRAacross['ba_combined']
comparisondf['CEBRA leave cohort out'] = CEBRAcoh['ba_combined']
g = sns.boxplot(data= comparisondf)
g.set(ylim=(0.49, 1))
g.axhline(0.5,ls='--')

## Calculate statistics
means = comparisondf.mean()
stds = comparisondf.std()