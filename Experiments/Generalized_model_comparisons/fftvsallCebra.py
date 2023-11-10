import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import random
import copy
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

leaveacross = True

suffix = '2023_10_10-09_54'

leave_cohort_out = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_cohort_out.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_within = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_sub_out_within_coh.npy"),
    allow_pickle="TRUE",
).item()
if leaveacross:
    leave_sub_across = np.load(
        os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix}_leave_1_sub_out_across_coh.npy"),
        allow_pickle="TRUE",
    ).item()
    listofdicts = [leave_cohort_out, leave_sub_across, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_across_cohorts", "leave_subject_within_cohort"]
else:
    listofdicts = [leave_cohort_out, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_within_cohort"]


Col = ['Cohort','Validation','Performance']
# Repeat this over
df = pd.DataFrame(columns=['Cohort','Validation','Performance'])
for cohort in ['Beijing','Pittsburgh','Berlin','Washington']:
    for valset in range(len(valtypes)):
        dataset = listofdicts[valset]
        result = [[cohort,valtypes[valset],dataset[cohort][item]['performance']] for item in dataset[cohort] if isinstance(dataset[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result,columns=Col)
        df = pd.concat([df,df_temp])

suffix_fft = '2023_10_10-12_04'

leave_cohort_out = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_fft}_leave_1_cohort_out.npy"),
    allow_pickle="TRUE",
).item()
leave_sub_within = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_fft}_leave_1_sub_out_within_coh.npy"),
    allow_pickle="TRUE",
).item()
if leaveacross:
    leave_sub_across = np.load(
        os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data\CEBRA performances", f"{suffix_fft}_leave_1_sub_out_across_coh.npy"),
        allow_pickle="TRUE",
    ).item()
    listofdicts = [leave_cohort_out, leave_sub_across, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_across_cohorts", "leave_subject_within_cohort"]
else:
    listofdicts = [leave_cohort_out, leave_sub_within]
    valtypes = ["leave_cohort_out", "leave_subject_within_cohort"]

Col = ['Cohort','Validation','Performance']
# Repeat this over
df_fft = pd.DataFrame(columns=['Cohort','Validation','Performance'])
for cohort in ['Beijing','Pittsburgh','Berlin','Washington']:
    for valset in range(len(valtypes)):
        dataset = listofdicts[valset]
        result = [[cohort,valtypes[valset],dataset[cohort][item]['performance']] for item in dataset[cohort] if isinstance(dataset[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result,columns=Col)
        df_fft = pd.concat([df_fft,df_temp])

#sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
sns.catplot(data=df, x="Validation", y="Performance", kind="box",color="0.9")
sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
              size=4, edgecolor="black",dodge=True).set_title(f"{suffix}")
plt.ylim(0.3, 1)
plt.title('All feature CEBRA')
plt.show()

sns.catplot(data=df_fft, x="Validation", y="Performance", kind="box",color="0.9")
sns.swarmplot(x="Validation", y="Performance", data=df_fft, hue="Cohort",
              size=4, edgecolor="black",dodge=True).set_title(f"{suffix_fft}")
plt.ylim(0.3, 1)
plt.title('fft only CEBRA')
plt.show()

df['Performance_fft'] = df_fft['Performance']

df_coh = df.loc[df['Validation'] == valtypes[0]]
data = pd.melt(df_coh[['Performance_fft','Performance']])
data['nr'] = list(range(39))+list(range(39))
plt.figure()
sns.boxplot(x="variable", y="value", data=data)
sns.swarmplot(x="variable", y="value", data=data, color=".25")
pal = sns.color_palette(['black'], df.shape[1])
sns.lineplot(x="variable", y="value", hue='nr', data=data,
             estimator=None, legend=False,palette=pal)

df['diff'] = df['Performance'] - df['Performance_fft']

## Analyze the mean difference per category
plt.figure()
df.groupby(['Cohort','Validation']).mean()['diff'].plot(kind='bar')
plt.title('Difference between CEBRA with all features vs fft')
plt.ylabel('CEBRA all - CEBRA fft BA')
plt.figure()
g = sns.boxplot(x="Cohort", y="diff", data=df,hue='Validation',showmeans=True,order=['Pittsburgh', 'Berlin', 'Beijing', 'Washington'],hue_order=['leave_subject_within_cohort',
                                                                                                                                                        'leave_subject_across_cohorts',
                                                                                                                                                        'leave_cohort_out'],
            meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"5"})
plt.plot([], [], 'o',markerfacecolor="white",markeredgecolor="black",markersize="5", linewidth=1, color='Crimson', label='mean')
plt.legend(fontsize="13")
g.set_title('Generalized, across-cohort decoding',fontsize=17)
plt.axhline(y=0, color='r', linestyle='--')
g.set_xlabel("Cohort",fontsize=17)
g.set_ylabel("Balanced accuracy: CEBRA all features - CEBRA fft",fontsize=17)
g.tick_params(labelsize=15)
#plt.savefig(r"C:\Users\ICN_GPU\Documents\Glenn_Data\Figures\PosterFigures\fftvsallCEBRA.pdf")

def permutationTest(x, y, plot_distr=True, x_unit='', p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    # Compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))

    pV = np.concatenate((x, y), axis=0)
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(np.abs(np.average(pS[0:int(len(pS)/2)]) - np.average(
            pS[int(len(pS)/2):])))

    # Calculate p-value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0])/p
    else:
        p_val = len(np.where(pD >= gT)[0])/p

    if plot_distr is True:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth "+x_unit+"="+str(gT)+" p="+str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val

def permutationTest_relative(x, y, plot_distr=True, x_unit='', p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    gT = np.abs(np.average(x) - np.average(y))
    pD = []
    for i in range(0, p):
        l_ = []
        for i in range(x.shape[0]):
            if random.randint(0, 1) == 1:
                l_.append((x[i], y[i]))
            else:
                l_.append((y[i], x[i]))
        pD.append(np.abs(np.average(np.array(l_)[:, 0]) - np.average(
            np.array(l_)[:, 1])))
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0])/p
    else:
        p_val = len(np.where(pD >= gT)[0])/p

    if plot_distr is True:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth "+x_unit+"="+str(gT)+" p="+str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()

    return gT, p_val

cohs = ['Beijing','Pittsburgh','Berlin','Washington']
vals = ['leave_cohort_out','leave_subject_across_cohorts','leave_subject_within_cohort']
gTlist = []
plist = []
p_permutepackage = []
condition = []
from permute.core import one_sample
for i in cohs:
    for j in vals:
        allfeatureperf = np.array(df.loc[np.array(df['Cohort'] == i) & np.array(df['Validation'] == j)]['Performance'])
        fftperf = np.array(df.loc[np.array(df['Cohort'] == i) & np.array(df['Validation'] == j)]['Performance_fft'])
        gT,p_val = permutationTest_relative(allfeatureperf,fftperf)
        (p, diff_means) = one_sample(allfeatureperf, fftperf, stat='mean',alternative='two-sided')
        p_permutepackage.append(p)
        gTlist.append(gT)
        plist.append(p_val)
        condition.append(i+','+j)

ax = plt.gca()
plt.bar(condition,plist)
ax.set_xticklabels(condition, rotation=45, ha='right')

gTlisttot = []
plisttot = []
p_permutepackagetot = []
conditiontot = []
for j in vals:
    allfeatureperf = np.array(df.loc[np.array(df['Validation'] == j)]['Performance'])
    fftperf = np.array(df.loc[np.array(df['Validation'] == j)]['Performance_fft'])
    gT,p_val = permutationTest_relative(allfeatureperf,fftperf)
    (p, diff_means) = one_sample(allfeatureperf, fftperf, stat='mean',alternative='two-sided')
    p_permutepackagetot.append(p)
    gTlisttot.append(gT)
    plisttot.append(p_val)
    conditiontot.append(j)

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"
