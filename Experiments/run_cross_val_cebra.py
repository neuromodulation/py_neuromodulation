import numpy as np
import os
import pandas as pd
import cebra
from cebra import CEBRA
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics, neighbors
from sklearn import linear_model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard.summary import hparams
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

ch_all = np.load(
    os.path.join(r"D:\Glenn", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
df_best_rmap = pd.read_csv(r"D:\Glenn\df_best_func_rmap_ch.csv")


cohorts = ["Beijing", "Pittsburgh", "Berlin", ]  # "Washington"

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
    ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
        "ch"
    ].iloc[0]
    X_test, y_test = get_data_sub_ch(
        ch_all, cohort_test, sub_test, ch_test
    )
    return X_test, y_test

def plot_results(perflist,val_approach, cohorts, save=False):
    listofdicts = perflist
    Col = ['Cohort', 'Validation', 'Performance']
    # Repeat this over
    df = pd.DataFrame(columns=['Cohort', 'Validation', 'Performance'])
    for cohort in cohorts:
        dataset = listofdicts
        result = [[cohort, val_approach, dataset[cohort][item]['performance']] for item in dataset[cohort]]
        df_temp = pd.DataFrame(data=result, columns=Col)
        df = pd.concat([df, df_temp])

    # sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
    sns.catplot(data=df, x="Validation", y="Performance", kind="box", color="0.9")
    sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
                  size=4, edgecolor="black", dodge=True)
    plt.ylim(0.3, 1)
    if save:
        writer.add_figure('Performance_Figure', plt.gcf(), 0)

def run_CV(val_approach,curtime,model_params):
    data_select = ch_all
    p_ = {}
    batotal = []
    bacohortlist = []
    cohort_prev_it = ""
    alllosses = 0
    alltemps = 0
    for cohort_test in cohorts:
        if cohort_test not in p_:
            p_[cohort_test] = {}
        bacohort = []
        for sub_test in data_select[cohort_test].keys():
            print('Val approach, cohort, subject:', val_approach, cohort_test, sub_test)
            if sub_test not in p_[cohort_test]:
                p_[cohort_test][sub_test] = {}
            X_test, y_test = get_data_channels(
                sub_test, cohort_test, df_rmap=df_best_rmap
            )

            # if statement to keep the same model for all subjects of leave 1 out cohort
            if (val_approach == "leave_1_cohort_out" and cohort_test != cohort_prev_it) or val_approach != "leave_1_cohort_out":
                cohorts_train = get_patients_train_dict(
                    sub_test, cohort_test, val_approach=val_approach, data_select=data_select
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

                # X_train, y_train, X_test, y_test = self.decoder.append_samples_val(X_train, y_train, X_test, y_test, 5)

                cebra_model = CEBRA(
                    model_architecture = model_params['model_architecture'], # previously used: offset1-model-v2'
                    batch_size = model_params['batch_size'],
                    temperature_mode=model_params['temperature_mode'],
                    learning_rate = model_params['learning_rate'],
                    max_iterations = model_params['max_iterations'],  # 50000
                    time_offsets = model_params['time_offsets'],
                    output_dimension = model_params['output_dimension'],
                    device = "cuda",
                    conditional='time_delta',
                    verbose = True
                )

                ### Add auxillary variables here
                if model_params['true_msess']:
                    if not model_params['pseudoDiscr']:
                        cebra_model.fit(X_train_comb, y_train_comb)
                    else:
                        cebra_model.fit(X_train_comb, np.array(y_train_comb, dtype=float))
                else:
                    if not model_params['pseudoDiscr']:
                        cebra_model.fit(X_train, y_train)
                    else: # Pretend the integer y_train is floating
                        cebra_model.fit(X_train, np.array(y_train, dtype=float), coh_aux)

                if model_params['true_msess']:
                    X_train_emb = cebra_model.transform(X_train_comb[0],session_id=0)
                else:
                    X_train_emb = cebra_model.transform(X_train, session_id=0)
                if model_params['all_embeddings']:
                    for i_emb in range(1,nr_embeddings):
                        X_train_emb = np.concatenate((X_train_emb, cebra_model.transform(X_train_comb[i_emb],session_id=i_emb)))

                # Get the loss and temperature plots
                if type(alllosses) == int:
                    alllosses = np.array(np.expand_dims(cebra_model.state_dict_["loss"],axis=0))
                    alltemps = np.array(np.expand_dims(cebra_model.state_dict_["log"]["temperature"],axis=0))
                else:
                    alllosses = np.concatenate((alllosses,np.expand_dims(cebra_model.state_dict_["loss"],axis=0)),axis=0)
                    alltemps = np.concatenate((alltemps, np.expand_dims(cebra_model.state_dict_["log"]["temperature"],axis=0)),axis=0)

                if model_params['decoder'] == 'KNN':
                    decoder = neighbors.KNeighborsClassifier(
                        n_neighbors=model_params['n_neighbors'], metric=model_params['metric'],
                        n_jobs=model_params['n_jobs'])
                elif model_params['decoder'] == 'Logistic':
                    decoder = linear_model.LogisticRegression(class_weight="balanced")
                elif model_params['decoder'] == 'SVM':
                    decoder = SVC(kernel=cosine_similarity)

                # Fitting is always done on the true discrete data
                decoder.fit(X_train_emb, np.array(y_train_discr, dtype=int))

            X_test_emb = cebra_model.transform(X_test,session_id=0)

            y_test_pr = decoder.predict(X_test_emb)
            ba = metrics.balanced_accuracy_score(y_test, y_test_pr)
            print(ba)
            # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(X_test_emb))

            # cebra.plot_embedding(embedding, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train_cont)
            p_[cohort_test][sub_test] = {}
            p_[cohort_test][sub_test]["performance"] = ba
            #p_[cohort_test][sub_test]["X_test_emb"] = X_test_emb
            p_[cohort_test][sub_test]["y_test"] = y_test
            p_[cohort_test][sub_test]["y_test_pr"] = y_test_pr
            p_[cohort_test][sub_test]["loss"] = cebra_model.state_dict_["loss"]
            p_[cohort_test][sub_test]["temp"] = cebra_model.state_dict_["log"]["temperature"]

            # Save some performance metrics
            bacohort.append(ba)
            batotal.append(ba)
            # Needed for training once per across cohort
            cohort_prev_it = cohort_test
        # After all subjects
        bacohortlist.append(np.mean(bacohort))
    # Save the output if it was not a debug run
    if not model_params['debug']:
        # After the run
        np.save(
            f"D:/Glenn/CEBRA performances/{curtime}_{val_approach}.npy",
            p_,
        )
        plot_results(p_, val_approach, cohorts, save=True)
        # Calculate some final performance metric and save to TensorBoard
        mean_ba = np.mean(batotal)
        metric_dict = {'mean_accuracy': mean_ba}
        # Calculate the mean of means (i.e. the performance mean ignoring imbalances in cohort sizes)
        ba_mean_ba = np.mean(bacohortlist)
        metric_dict['cohortbalanced_mean_accuracy'] = ba_mean_ba
        for coh in range(len(cohorts)):
            metric_dict[f'mean_{cohorts[coh]}'] =  bacohortlist[coh]
        print(metric_dict)
        exp, ssi, sei = hparams(model_params, metric_dict)
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            writer.add_scalar(k, v)
        medloss = np.median(alllosses, axis=0)
        medtemp = np.median(alltemps, axis=0)
        for it in range(model_params['max_iterations']):
            writer.add_scalar('Median_loss',medloss[it],it)
            writer.add_scalar('Median_temp',medtemp[it],it)

# In time change this setup to use PYTORCH and TENSORBOARD to keep track of iterations, model params and output
curtime = datetime.now().strftime("%Y_%m_%d-%H_%M")
experiment = "All_channels"
longcompute = "leave_1_sub_out_across_coh"
perflist = []
val_approaches = ["leave_1_cohort_out"]#, "leave_1_sub_out_within_coh"]


for val_approach in val_approaches:
    model_params = {'model_architecture':'offset10-model',
                'batch_size': 512, # Ideally as large as fits on the GPU, min recommended = 512
                'temperature_mode':"auto",
                'learning_rate': 0.005, # Set this in accordance to loss function progression in TensorBoard
                'max_iterations': 3000,  # 50000, Set this in accordance to the loss functions in TensorBoard
                'time_offsets': 1, # Time offset between samples (Ideally set larger than receptive field according to docs)
                'output_dimension': 3, # Nr of output dimensions of the CEBRA model
                'decoder': 'SVM', # Choose from "KNN" or "Logistic"
                'n_neighbors': 3, # KNN setting
                'metric': "cosine", # KNN setting
                'n_jobs': 20, # KNN setting
                'all_embeddings':False, # If you want to combine all the embeddings (only when true_msess = True !), currently 1 model is used for the test set --> Make majority
                'true_msess':False, # Make a single model will be made for every subject (Usefull if feature dimension different between subjects)
                'discreteMov':False, # Turn pseudoDiscr to False if you want to test true discrete movement labels (Otherwise this setting will do nothing)
                'pseudoDiscr': True, # Pseudodiscrete meaning direct conversion from int to float
                'gaussSigma':1.5, # Set pseuodDiscr to False for this to take effect and assuming a Gaussian for the movement distribution
                'additional_comment':'Cohort_auxillary',
                'debug': True} # Debug = True; stops saving the results unnecessarily
    if not model_params['debug']:
        writer = SummaryWriter(log_dir=f"D:\Glenn\CEBRA_logs\{val_approach}\{curtime}")

    run_CV(val_approach, curtime, model_params)

### Code improvements
# TODO: Look into implementing a K-fold strategy (i.e. leave 2 out instead of 1, especially for across cohort to speed up without much reduced training set)
# TODO: ? Save the standard deviations as well.
# TODO: ? Would a loss plot per cohort be better (assuming that there are differences between cohorts)
### Tests to perform
# DONE: Test the difference between using a discrete auxillary, no auxillary and a continuous auxillary
# TODO: Check what happens upon changing the way auxillary is made continuous (AMPLITUDE & SIGMA)
# DONE: Investigate adding subject information as discrete auxillary
# TODO: Look into adding multiple channels (either as feature dim, or concatenated with labels)
# TODO: Test different model architectures (with 2dConv to integrate info over the channels?)
# TODO: --> Might require the loader to supply data differently (Or make a custom implementation)
# TODO: Look at the embeddings (In single session and in multi-session cases
# TODO: Implement FAISS KNN
# TODO: Look at dataset size / distribution and (no)movement distribution


# TODO: Embedding for each channel & Use Class. score as weight for the embeddings (???) Which model for test then ? --> Correlation to R-map could be used
# TODO: Seems like a good idea, but need A LOT of compute

# TODO: Idea is, by sampling uniformly it becomes invariant to that; might add UPDRS score in a way
### Findings
# Psuedodiscrete auxillary works as good or better than the gaussian filtered one, but full discrete does not work well
# Preliminary findings: Including subject ID makes the performance worse for all datasets except marginally Pittsburgh