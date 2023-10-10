import numpy as np
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics, neighbors
from sklearn import linear_model
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import plotly
import plotly.graph_objects as go
import plotly.express as px
import xgboost
from sklearn.utils import class_weight
from Experiments.utils.knn_bpp import kNN_BPP
# from einops.layers.torch import Rearrange # --> Can add to the model to reshape to fit 2dConv maybe

ch_all_train = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

ch_all_test = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "test_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()

ch_all_feat = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "TempCleaned2_channel_all.npy"), # TempCleaned2_channel_all.npy
    allow_pickle="TRUE",
).item()
df_best_rmap = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch_adapted.csv")
df_best_rmap_old = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch.csv")
df_updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")

features = ['Hjorth','fft', 'Sharpwave', 'fooof', 'bursts']
performances = []
featuredim = ch_all_feat['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
featuredict = {}
for i in range(len(features)):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    featuredict[features[i]] = idx_i



# offset9 should be better than offset 10, as the selected index will always cause class majority in the receptive field
# In contrast to offset 10 where equality between classes can be had if selected value on edge
# TODO: Select receptive field based on statistics of the dataset

def get_patients_train_dict(sub_test, cohort_test, val_approach: str, data_select: dict):
    cohorts_train = {}
    for cohort in cohorts:
        if (val_approach == "leave_1_cohort_out"
                and cohort == cohort_test):
            continue # Skips current iteration (i.e. do not incl. test cohort) trains on all other cohort data
        if (
            val_approach == "leave_1_sub_out_within_coh"
            and cohort != cohort_test
        ):
            continue # Only include test_cohort
        cohorts_train[cohort] = []
        for sub in data_select[cohort]:
            if (
                val_approach == "leave_1_sub_out_within_coh"
                and sub == sub_test
                and cohort == cohort_test
            ):
                continue # Do not include test subject (trained on subjects from same cohort
            if (
                val_approach == "leave_1_sub_out_across_coh"
                and sub == sub_test
            ):
                continue # Do not include test subject (but trained on all cohorts and other subject)
            cohorts_train[cohort].append(sub)
    return cohorts_train

def get_data_sub_ch(channel_all, cohort, sub, ch):

    X_train = []
    y_train = []



    for f in channel_all[cohort][sub][ch].keys():
        X_train.append(channel_all[cohort][sub][ch][f]["data"])
        y_train.append(channel_all[cohort][sub][ch][f]["label"])

    if len(X_train) > 1:
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

    else:
        X_train = X_train[0]
        y_train = y_train[0]

    return X_train, y_train

def get_data_channels(sub_test: str, cohort_test: str, df_rmap: list):
    if sub_test != '000':
        sub_num = sub_test.lstrip('0')
    else:
        sub_num = '0'
    ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_num")[
        "ch"
    ].iloc[0]
    X_test, y_test = get_data_sub_ch(
        ch_all_feat, cohort_test, sub_test, ch_test
    )
    return X_test, y_test

def plot_results(perflist,val_approach, cohorts, save=False):
    Col = ['Cohort', 'Validation', 'Performance']
    # Repeat this over
    df = pd.DataFrame(columns=['Cohort', 'Validation', 'Performance'])
    for cohort in cohorts:
        result = [[cohort, val_approach, perflist[cohort][item]['performance']] for item in perflist[cohort] if isinstance(perflist[cohort][item],dict)]
        df_temp = pd.DataFrame(data=result, columns=Col)
        df = pd.concat([df, df_temp])

    # sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
    sns.catplot(data=df, x="Validation", y="Performance", kind="box", color="0.9")
    sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
                  size=4, edgecolor="black", dodge=True)
    plt.ylim(0.3, 1)
    #plt.show()
    if save:
        writer.add_figure('Performance_Figure', plt.gcf(), 0)


model = linear_model.LogisticRegression(class_weight="balanced", C = 1,max_iter=1000) # C does not seem to have an effect
def run_CV(val_approach,curtime,model_params,show_embedding=False,Testphase=False):
    train_select = ch_all_train
    test_select = ch_all_test
    p_ = {}
    batotal = []
    bacohortlist = []
    cohort_prev_it = ""

    for cohort_test in cohorts:
        if cohort_test not in p_:
            p_[cohort_test] = {}
        bacohort = []
        if Testphase:
            subtests = test_select[cohort_test].keys()
        else:
            subtests = train_select[cohort_test].keys()
        for sub_test in subtests:
            print('Val approach, cohort, subject:', val_approach, cohort_test, sub_test)
            if sub_test not in p_[cohort_test]:
                p_[cohort_test][sub_test] = {}
            X_test, y_test = get_data_channels(
                sub_test, cohort_test, df_rmap=df_best_rmap
            )
            # Select the desired features
            toselect = model_params['features'].split(',')
            idxlist = []
            for featsel in toselect:
                idxlist.append(featuredict[featsel])
            idxlist = np.concatenate(idxlist)
            X_test = X_test[:, idxlist].copy()

            # if statement to keep the same model for all subjects of leave 1 out cohort
            if (val_approach == "leave_1_cohort_out" and cohort_test != cohort_prev_it) or val_approach != "leave_1_cohort_out":
                if not Testphase:
                    cohorts_train = get_patients_train_dict(
                        sub_test, cohort_test, val_approach=val_approach, data_select=ch_all_train # Use train/val split in training (except test data, which gets rejected in the function)
                    )
                else:
                    cohorts_train = get_patients_train_dict(
                        sub_test, cohort_test, val_approach=val_approach, data_select=ch_all
                        # Use all data in training (except test data, which gets rejected in the function)
                    )

                X_train_comb = []
                y_train_comb = []
                y_train_discr_comb = []
                coh_aux_comb = []
                sub_aux_comb = []
                nr_embeddings = 0
                sub_counter = 0
                coh_counter = 0
                for cohort_train in list(cohorts_train.keys()):
                    for sub_train in cohorts_train[cohort_train]:
                        nr_embeddings += 1
                        X_train, y_train = get_data_channels(
                            sub_train, cohort_train, df_rmap=df_best_rmap
                        )
                        sub_aux = np.tile(sub_counter,len(y_train))

                        y_train_discr_comb.append(np.squeeze(y_train)) # Save the true labels
                        if (not model_params['discreteMov']) and (not model_params['pseudoDiscr']):
                            y_train = gaussian_filter1d(np.array(y_train, dtype=float),
                                                        sigma=model_params['gaussSigma'])
                        X_train_comb.append(np.squeeze(X_train))
                        y_train_comb.append(np.squeeze(y_train))
                        sub_aux_comb.append(np.squeeze(sub_aux))
                        sub_counter += 1
                        coh_aux = np.tile(coh_counter, len(y_train))
                        coh_aux_comb.append(np.squeeze(coh_aux))
                    coh_counter += 1

                if len(X_train_comb) > 1:
                    X_train = np.concatenate(X_train_comb, axis=0)
                    y_train = np.concatenate(y_train_comb, axis=0)
                    y_train_discr = np.concatenate(y_train_discr_comb, axis=0)
                    sub_aux = np.concatenate(sub_aux_comb, axis=0)
                    coh_aux = np.concatenate(coh_aux_comb, axis=0)
                else:
                    X_train = X_train_comb[0]
                    y_train = X_train_comb[0]
                    y_train_discr = X_train_comb[0]
                    sub_aux = sub_aux_comb[0]
                    coh_aux = coh_aux_comb[0]
                #print(X_train.shape[0])
                X_train = X_train[:,idxlist].copy()
                #print(X_test.shape[0])
                # Proof that TEST not in Train
                #def is_a_in_x(A, X):
                #    for i in range(len(X) - len(A) + 1):
                #        if all(A == X[i:i + len(A)]):
                #            return True
                #    return False
                #print(is_a_in_x(X_test[:,0],X_train[:,0]))
                # X_train, y_train, X_test, y_test = self.decoder.append_samples_val(X_train, y_train, X_test, y_test, 5)

                # Fit the logistic regression
                decoder = model.fit(X_train,y_train_discr)
            # Predict with the logstic regressor
            y_test_pr = decoder.predict(X_test)
            ba = metrics.balanced_accuracy_score(y_test, y_test_pr)
            #confusion = metrics.confusion_matrix(y_test,y_test_pr)
            print(ba)
            #print(confusion)
            # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(X_test_emb))

            # cebra.plot_embedding(embedding, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train_cont)
            p_[cohort_test][sub_test] = {}
            p_[cohort_test][sub_test]["performance"] = ba
            #p_[cohort_test][sub_test]["X_test_emb"] = X_test_emb
            p_[cohort_test][sub_test]["y_test"] = y_test
            p_[cohort_test][sub_test]["y_test_pr"] = y_test_pr

            # Save some performance metrics
            bacohort.append(ba)
            batotal.append(ba)

        # After all subjects
        bacohortlist.append(np.mean(bacohort))
        mean_ba = np.mean(batotal)
        print(f'running mean balanced accuracy: {mean_ba}')
        p_[cohort_test]['mean_ba'] = np.mean(bacohort)
        p_[cohort_test]['std_ba'] = np.std(bacohort)
    # Save the output if it was not a debug run
    if not model_params['debug']:
        p_['total_mean_ba'] = mean_ba
        p_['total_std'] = np.std(batotal)
        # After the run
        np.save(
            f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA performances/{curtime}_{val_approach}.npy",
            p_,
        )
        plot_results(p_, val_approach, cohorts, save=True)
        metric_dict = {'mean_accuracy': mean_ba}
        # Calculate the mean of means (i.e. the performance mean ignoring imbalances in cohort sizes)
        ba_mean_ba = np.mean(bacohortlist)
        metric_dict['cohortbalanced_mean_accuracy'] = ba_mean_ba
        for coh in range(len(cohorts)):
            metric_dict[f'mean_{cohorts[coh]}'] = bacohortlist[coh]
        exp, ssi, sei = hparams(model_params, metric_dict)
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            writer.add_scalar(k, v)
        #medloss = np.median(alllosses, axis=0)
        #medtemp = np.median(alltemps, axis=0)
        #for it in range(model_params['max_iterations']):
        #    writer.add_scalar('Median_loss', medloss[it], it)
        #    writer.add_scalar('Median_temp', medtemp[it], it)

# In time change this setup to use PYTORCH and TENSORBOARD to keep track of iterations, model params and output
curtime = datetime.now().strftime("%Y_%m_%d-%H_%M")
experiment = "All_channels"
longcompute = "leave_1_sub_out_across_coh"
perflist = []
val_approaches = ["leave_1_sub_out_within_coh","leave_1_cohort_out","leave_1_sub_out_across_coh"]
cohorts = [ "Pittsburgh","Beijing", "Berlin",'Washington']

for val_approach in val_approaches:
    model_params = {'model_architecture':'LogisticReg',
                'batch_size': 512, # Ideally as large as fits on the GPU, min recommended = 512
                'temperature_mode':'auto', # Constant or auto
                'temperature':1,
                'min_temperature':0.1, # If temperature mode = auto this should be set in the desired expected range
                'learning_rate': 0.005, # Set this in accordance to loss function progression in TensorBoard
                'max_iterations': 100,  # 5000, Set this in accordance to the loss functions in TensorBoard
                'time_offsets': 1, # Time offset between samples (Ideally set larger than receptive field according to docs)
                'output_dimension': 3, # Nr of output dimensions of the CEBRA model
                'decoder': 'Logistic', # Choose from "KNN", "Logistic", "SVM", "KNN_BPP"
                'n_neighbors': 35, # KNN & KNN_BPP setting (# of neighbours to consider) 35 works well for 3 output dimensions
                'metric': "euclidean", # KNN setting (For L2 normalized vectors, the ordering of Euclidean and Cosine should be the same)
                'n_jobs': 20, # KNN setting for parallelization
                'all_embeddings':False, # If you want to combine all the embeddings (only when true_msess = True !), currently 1 model is used for the test set --> Make majority
                'true_msess':False, # Make a single model will be made for every subject (Usefull if feature dimension different between subjects)
                'discreteMov':True, # Turn pseudoDiscr to False if you want to test true discrete movement labels (Otherwise this setting will do nothing)
                'pseudoDiscr': False, # Pseudodiscrete meaning direct conversion from int to float
                'gaussSigma':1.5, # Set pseuodDiscr to False for this to take effect and assuming a Gaussian for the movement distribution
                'features': 'Hjorth,fft,Sharpwave,fooof,bursts',# Choose what features to include 'Hjorth,fft,Sharpwave,fooof,bursts' as 1 string separated by commas without spaces
                'additional_comment':'Val_all_Logres',
                'debug': False} # Debug = True; stops saving the results unnecessarily
    if not model_params['debug']:
        writer = SummaryWriter(log_dir=f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA_logs/{val_approach}/{curtime}")

    run_CV(val_approach, curtime, model_params,show_embedding=False,Testphase=False)