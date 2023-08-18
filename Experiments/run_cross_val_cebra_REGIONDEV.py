import numpy as np
import os
import pandas as pd
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
import plotly
import plotly.graph_objects as go
import plotly.express as px
import xgboost
from sklearn.utils import class_weight
from Experiments.utils.knn_bpp import kNN_BPP

ch_all = np.load(
    os.path.join(r"D:\Glenn", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
df_full = pd.read_csv(r"D:\Glenn\df_ch_performances_regions.csv")


cohorts = ["Beijing", "Pittsburgh", "Berlin", ]  # "Washington"
Atlas = 'DiFuMo128'
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

def get_data_channels(sub_test: str, cohort_test: str, df_all, test: bool, model_params: dict):
    # TODO: seperate if statement for test data --> Use all channels in brain region OR best R-map OR concat of # R-maps
    ### Is it fair to select best single channel performers for the regions
    brain_aux = [] # In order for it to exist always and return, but is empty when not needed
    if model_params['Regions']: # Check if something in the list --> Use regions
        if test: # If test select all channels from the brain regions
            X_test = []
            y_test = []
            for regionit in range(len(model_params['Regions'])):  # Go through the regions and put in separate dimension
                X_test_reg = []
                y_test_reg = []
                curregion = model_params['Regions'][regionit]
                ch_test_list = df_all.query(f"cohort == @cohort_test and sub == @sub_test and {Atlas} == @curregion")[
                    "ch"]
                for i in range(len(ch_test_list)): # For test select all channels
                    ch_test = ch_test_list.iloc[i]

                    X_test_temp, y_test_temp = get_data_sub_ch(
                        ch_all, cohort_test, sub_test, ch_test
                    )
                    X_test_reg.append(X_test_temp)
                    y_test_reg.append(y_test_temp)
                # Not sure of this is needed or not but it should concat the mult. channel per region
                if len(X_test) > 1:
                    X_test_reg = np.concatenate(X_test_reg, axis=0)
                    y_test_reg = np.concatenate(y_test_reg, axis=0)
                else:
                    X_test_reg = X_test_reg[0]
                    y_test_reg = y_test_reg[0]
                # Append regions to full list --> dimension of # brain regions, containing their signal
                X_test.append(X_test_reg)
                y_test.append(y_test_reg)
            if model_params['TimeConcat']:  # Remove the region dimension and concatenate (also create region aux)
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)
                else:
                    X_test = X_test[0]
                    y_test = y_test[0]
        else:
            X_test = []
            y_test = []
            for regionit in range(len(model_params['Regions'])): # Go through the regions and put in separate dimension
                X_test_reg = []
                y_test_reg = []
                curregion = model_params['Regions'][regionit]
                ch_test_list = df_all.query(f"cohort == @cohort_test and sub == @sub_test and {Atlas} == @curregion").sort_values(
                    by='performance', ascending=False)[
                    "ch"]
                for i in range(model_params['nrchannels'][regionit]):  # Loop over how many channels
                    ch_test = ch_test_list.iloc[i]

                    X_test_temp, y_test_temp = get_data_sub_ch(
                        ch_all, cohort_test, sub_test, ch_test
                    )
                    X_test_reg.append(X_test_temp)
                    y_test_reg.append(y_test_temp)
                # Not sure of this is needed or not but it should concat the mult. channel per region
                if len(X_test) > 1:
                    X_test_reg = np.concatenate(X_test_reg, axis=0)
                    y_test_reg = np.concatenate(y_test_reg, axis=0)
                else:
                    X_test_reg = X_test_reg[0]
                    y_test_reg = y_test_reg[0]
                # Make a brain region auxillary variable
                if model_params['TimeConcat']:
                    brain_aux.append(np.repeat(regionit, len(X_test_reg)))
                # Append regions to full list --> dimension of # brain regions, containing their signal
                X_test.append(X_test_reg)
                y_test.append(y_test_reg)
            if model_params['TimeConcat']: # Remove the region dimension and concatenate (also create region aux)
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)
                    brain_aux = np.concatenate(brain_aux, axis=0)
                else:
                    X_test = X_test[0]
                    y_test = y_test[0]
                    brain_aux = brain_aux[0]

    else: # No regions were supplied --> Use R-Map
        if test:
            if model_params['TimeConcat']:  # Append multiple R-maps according to Nrchannels
                X_test = []
                y_test = []
                ch_test_list = \
                df_all.query("cohort == @cohort_test and sub == @sub_test").sort_values(by='r_func', ascending=False)[
                    "ch"]
                for i in range(model_params['nrtestR']):  # Loop over how many R-maps we want to concatenate
                    ch_test = ch_test_list.iloc[i]

                    X_test_temp, y_test_temp = get_data_sub_ch(
                        ch_all, cohort_test, sub_test, ch_test
                    )
                    X_test.append(X_test_temp)
                    y_test.append(y_test_temp)
                # Not sure of this is needed or not but it should concat them
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)

                else:
                    X_test = X_test[0]
                    y_test = y_test[0]
            else:  # 1-Channel R-map case
                ch_test = \
                df_all.query("cohort == @cohort_test and sub == @sub_test").sort_values(by='r_func', ascending=False)[
                    "ch"].iloc[0]

                X_test, y_test = get_data_sub_ch(
                    ch_all, cohort_test, sub_test, ch_test
                )
        else:
            if model_params['TimeConcat']: # Append multiple R-maps according to Nrchannels
                X_test = []
                y_test = []
                ch_test_list = df_all.query("cohort == @cohort_test and sub == @sub_test").sort_values(by='r_func', ascending=False)[
                    "ch"]
                for i in range(model_params['nrchannels'][0]): # Loop over how many R-maps we want to concatenate
                    ch_test = ch_test_list.iloc[i]

                    X_test_temp, y_test_temp = get_data_sub_ch(
                        ch_all, cohort_test, sub_test, ch_test
                    )
                    X_test.append(X_test_temp)
                    y_test.append(y_test_temp)
                # Not sure of this is needed or not but it should concat them
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)

                else:
                    X_test = X_test[0]
                    y_test = y_test[0]
            else: # 1-Channel R-map case
                ch_test = df_all.query("cohort == @cohort_test and sub == @sub_test").sort_values(by='r_func', ascending=False)[
                    "ch"].iloc[0]

                X_test, y_test = get_data_sub_ch(
                    ch_all, cohort_test, sub_test, ch_test
                )
        return X_test, y_test, brain_aux

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

def plotly_embeddings(X_train_emb,X_test_emb,y_train_discr,y_test,aux=None,type='none'):
#    Plot together with the test embedding (For coh auxillary)
    symbollist = ['circle', 'cross', 'diamond', 'square', 'x']
    if type == 'coh':
        colorlist = px.colors.qualitative.D3
        plotly.offline.plot({'data': [go.Scatter3d(
            x=np.append(X_train_emb[:, 0],X_test_emb[:, 0]),
            y=np.append(X_train_emb[:, 1],X_test_emb[:, 1]),
            z=np.append(X_train_emb[:, 2],X_test_emb[:, 2]),
            mode='markers',
            marker=dict(
                size=2,
                color=np.array(colorlist)[np.array(np.append(y_train_discr,np.array(y_test,dtype=int)+2), dtype=int)],
                symbol=np.array(symbollist)[np.array(np.append(aux,np.repeat(max(aux)+1,len(y_test))), dtype=int)],
                # set color to an array/list of desired values
                opacity=0.8))]}, auto_open=True)
    elif type == 'sub':
        colorlist = px.colors.qualitative.Light24  # Light24 might work for sub (# of colours) and D3 works well for coh_aux
        plotly.offline.plot({'data': [go.Scatter3d(
            x=np.append(X_train_emb[:, 0],X_test_emb[:, 0]),
            y=np.append(X_train_emb[:, 1],X_test_emb[:, 1]),
            z=np.append(X_train_emb[:, 2],X_test_emb[:, 2]),
            mode='markers',
            marker=dict(
                size=2,
                color=np.array(colorlist)[np.array(np.append(aux,np.repeat(max(aux)+1,len(y_test))), dtype=int)],
                symbol=np.array(symbollist)[np.array(np.append(y_train_discr,np.array(y_test,dtype=int)), dtype=int)],
                # set color to an array/list of desired values
                opacity=0.8))]}, auto_open=True)
    elif type == 'none':
        colorlist = px.colors.qualitative.D3
        plotly.offline.plot({'data': [go.Scatter3d(
            x=np.append(X_train_emb[:, 0], X_test_emb[:, 0]),
            y=np.append(X_train_emb[:, 1], X_test_emb[:, 1]),
            z=np.append(X_train_emb[:, 2], X_test_emb[:, 2]),
            mode='markers',
            marker=dict(
                size=2,
                color=np.array(colorlist)[np.array(np.append(y_train_discr, np.array(y_test, dtype=int) + 2), dtype=int)],
                symbol=np.array(symbollist)[np.array(np.append(np.repeat(0,len(y_train_discr)), np.repeat(1, len(y_test))), dtype=int)],
                # set color to an array/list of desired values
                opacity=0.8))]}, auto_open=True)

def run_CV(val_approach,curtime,model_params,show_embedding=False):
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
            X_test, y_test, brain_aux = get_data_channels(
                sub_test, cohort_test, df_all=df_full, test=True, model_params=model_params
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
                        X_train, y_train, brain_aux = get_data_channels(
                            sub_train, cohort_train, df_all=df_full, test=False, model_params=model_params
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
                    distance='cosine',
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
                        cebra_model.fit(X_train, np.array(y_train,dtype=float),coh_aux)

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
                elif model_params['decoder'] == 'FAISS':
                    print('Not implemented')
                elif model_params['decoder'] == 'MLP':
                    print('Not implemented')
                elif model_params['decoder'] == 'XGB':
                    decoder = xgboost.sklearn.XGBClassifier()
                    #decoder.set_params(**{'lambda':2})
                    classes_weights = class_weight.compute_sample_weight(
                        class_weight="balanced", y=y_train_discr
                    )
                    decoder.set_params(eval_metric="logloss")
                    decoder.fit(
                        X_train_emb,
                        y_train_discr,
                        sample_weight=classes_weights,
                    )
                elif model_params['decoder'] == 'KNN_BPP':
                    decoder = kNN_BPP(n_neighbors=model_params['n_neighbors'])


                # Fitting of classifier is always done on the true discrete data
                decoder.fit(X_train_emb, np.array(y_train_discr, dtype=int))

            # Needed for training once per across cohort
            cohort_prev_it = cohort_test
            # TEST PERMUTATION OF FEATURES
            # rng = np.random.default_rng()
            # X_test_emb = cebra_model.transform(rng.permutation(X_test,axis=1),session_id=0)

            X_test_emb = cebra_model.transform(X_test,session_id=0)

            ### Embeddings are plotted here
            if show_embedding:
                plotly_embeddings(X_train_emb,X_test_emb,y_train_discr,y_test,aux=coh_aux,type='coh')

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

        # After all subjects
        bacohortlist.append(np.mean(bacohort))
        mean_ba = np.mean(batotal)
        print(f'running mean balanced accuracy: {mean_ba}')
    # Save the output if it was not a debug run
    if not model_params['debug']:
        # After the run
        np.save(
            f"D:/Glenn/CEBRA performances/{curtime}_{val_approach}.npy",
            p_,
        )
        plot_results(p_, val_approach, cohorts, save=True)
        # Calculate some final performance metric and save to TensorBoard
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
    model_params = {
        'model_architecture':'offset10-model',
        'batch_size': 512, # Ideally as large as fits on the GPU, min recommended = 512
        'temperature_mode':'auto', # Constant or auto
        'temperature':1,
        'min_temperature':0.1, # If temperature mode = auto this should be set in the desired expected range
        'learning_rate': 0.005, # Set this in accordance to loss function progression in TensorBoard
        'max_iterations': 5000,  # 5000, Set this in accordance to the loss functions in TensorBoard
        'time_offsets': 1, # Time offset between samples (Ideally set larger than receptive field according to docs)
        'output_dimension': 3, # Nr of output dimensions of the CEBRA model
        'decoder': 'KNN_BPP', # Choose from "KNN", "Logistic", "KNN_BPP"
        'n_neighbors': 35, # KNN & KNN_BPP setting (# of neighbours to consider) 35 works well for 3 output dimensions
        'metric': "euclidean", # KNN setting (For L2 normalized vectors, the ordering of Euclidean and Cosine should be the same)
        'n_jobs': 20, # KNN setting for parallelization
        'all_embeddings':False, # If you want to combine all the embeddings (only when true_msess = True !), currently 1 model is used for the test set --> Make majority
        'true_msess':False, # Make a single model will be made for every subject (Usefull if feature dimension different between subjects)
        'discreteMov':False, # Turn pseudoDiscr to False if you want to test true discrete movement labels (Otherwise this setting will do nothing)
        'pseudoDiscr': True, # Pseudodiscrete meaning direct conversion from int to float
        'gaussSigma':1.5, # Set pseuodDiscr to False for this to take effect and assuming a Gaussian for the movement distribution
        'Regions': [], # Set (list of) desired regions to analyze OR leave empty if you do not want to use regions (uses best R-map as default)
        'nrchannels': [2], # Nr of channels to use in train for each region (based on own analysis for balance) OR for in combination with TimeConcat=True & R-Map, how many channels in train + test
        'TimeConcat': True, # Set True if you want to either combine the regions in 1 embedding over time (with auxillary); or use multiple R-map channels
        'nrtestR': 1, # In case of TimeConcat True and Regions empty --> Set how many test channels you want to use (ordered on R-map corr)
        'additional_comment': 'Cohort_auxillary',
        'debug': True # Set to True will stop saving model outputs
        }
    if not model_params['debug']:
        writer = SummaryWriter(log_dir=f"D:\Glenn\CEBRA_logs\{val_approach}\{curtime}")

    run_CV(val_approach, curtime, model_params,show_embedding=False)

# Note: 2 CEBRA models trained to convergence should be the same up to a linear transformation (given enough data)

### Code improvements
##### 14/08:
# DONE: Look at the embeddings (In single session and in multi-session cases, and with different auxillary variables
##### 15/08:
# DONE: Implement bagging KNN with cosine metric --> Not needed as for L2 normalized vectors the ordering of Euclidean and Cosine should be the same
# DONE: Test what happens upon shuffling the test set features
##### 16/08:
# DONE: Compute brain regions (Using Thomas's code) and did some analysis
#### 17/08:
# UNFINISHED: Continue working on brain region model; Check proper selection to not throw away full cohorts
#   Multiple ideas: Either just select brain regions that >x% of the subject have electrode in and then add region as aux
#                   --> Lets the model itself choose a best match for each sample and map
#                   Could potentially include some imbalance in channels per brain regions (aux might balance sampling) ?
#                   OR: Embedding per brain region, would probably want more balance, at least per cohort
# TODO: (Look at dataset size / distribution (time axis) and (no)movement distribution)
#### 18/08:
# DONE: Run DiFuMo 64, 128 & 256 and analyze subject distribution
# DONE: Adapt this code (run_cross_val_cebra) to be able to run, concat. brain regions with aux var & model per brain region
#       --> Which channels to select (if choice), random vs highest performance ones (in train), in test prob. all available channels.
# DONE: Expand the Train_Val_Test_Split script to handle multi channel per subject
#### 19/08:
# TODO: (Look at dataset size / distribution (time axis) and (no)movement distribution)
# TODO: !!!! Implement that regions are unpacked before running in multi-region runs and create separate embeddings, decoders and test separately !!!
# GENERAL:
### Code improvements
# TODO: Look into implementing a K-fold strategy (i.e. leave 2 out instead of 1, especially for across cohort to speed up without much reduced training set)
# TODO: ? Save the standard deviations as well.
# TODO: ?? Would a loss plot per cohort be better (assuming that there are differences between cohorts)

### Tests to perform
# TODO: MULTI-CHANNEL IDEAS:
#   1. -Train: True Multi-session with all (or top x) channels as SEPARATE sessions
#       --> Class model per embedding (weigh embeddings by Corr to R-Map (or just channel R performance))
#      -Test: ?
#      -Problems: What model and channels do you use from the test subject / A LOT OF COMPUTE NEEDED
#   2. -Train: True Multi-session per brain region (diff channels as features) --> Class. model per brain region (and subject which can be combined for the class model)
#      -Test: Apply brain region models to channels in respective brain regions, and (?weighted R-map?) Majority vote
#      -Problems: Problem when test subject diff # of channels in brain region w.r.t. all train subjects --> No model available / Also some COMPUTE
#   3. -Train: Single-session per brain region (diff. channels over time axis) --> Class. model per brain region
#      -Test: Apply brain region models to channels in respective brain regions
#               Combine: Based on average brain region - R score single channel) or using val set to test best combi
#      -Problems: Having more channels in brain region will bias the sampling towards that subject
#      -Solution: Auxillary label per brain ID (and maybe combined with cohort?)
#   4. -Train: Single-session with top x correlators to R-map (either as feature (ordered) or over time)
#   5. -Train: Single-session with all (or most) channels sorted in feature dimension on their correlation to the R-map (Might provide some structure for the model to expect)
# TODO: Test different model architectures (with 2dConv to integrate info over the channels?)
# TODO: Include more features
# TODO: Hyperparameter search
# TODO: Look into performance difference treating as single session vs TRUE multisession (+ combining embeddings)

# DONE: Test the difference between using a discrete auxillary, no auxillary and a continuous auxillary
# TODO: Check what happens upon changing the way auxillary is made continuous (AMPLITUDE & SIGMA) (pot. not that useful)
# DONE: Investigate adding subject information as discrete auxillary
# TODO: Test different model architectures (with 2dConv to integrate info over the channels?)
#   --> Might require the loader to supply data differently (Or make a custom implementation)

# TODO: Inclusion of UPDRS-III ?

# QUESTION: How much is R-MAP correlation correlated to R-score. Maybe a machine learning model to predict R score based on (resting state) signal + R-map corr + brain region
#   might be better than only using the R-MAP

### Findings
# Psuedodiscrete auxillary works as good or better than the gaussian filtered one, but full discrete does not work well
# Preliminary findings: Including subject ID makes the performance worse for all datasets except marginally Pittsburgh

# KNN_BPP Can work almost as well as Logregression (but still a bit worse). Perhaps for higher dimension useful.
# XGB was also not that bad (Have to try more)

# The test embedding IS dependent on the ordering of the features, and thus will also be on the ordering of channels potentially
# Therefore, some ordering here is required, or look into true multisess / time concatenation

# Combining cohort and sub as cont. auxillary with movement as discrete does not work well. In general movement as discrete works a bit worse.
# NEED to try: Both cohort and movement as continuous.

# Try cohort as multisess (1 sess per cohort)