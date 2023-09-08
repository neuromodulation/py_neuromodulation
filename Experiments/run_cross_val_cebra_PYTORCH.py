import numpy as np
import os
import pandas as pd
from torch import nn
import torch
import cebra.models
import cebra.data
import cebra.distributions
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin, cebra_layers
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
# from einops.layers.torch import Rearrange # --> Can add to the model to reshape to fit 2dConv maybe
from Experiments.utils.cebracustom import CohortDiscreteDataLoader
import time
torch.backends.cudnn.benchmark = True

ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "train_channel_all_fft.npy"),
    allow_pickle="TRUE",
).item()
ch_all_feat = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_noraw.npy"),
    allow_pickle="TRUE",
).item()
df_best_rmap = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch.csv")


cohorts = ["Beijing", "Pittsburgh", "Berlin"] # "Washington"]

# offset9 should be better than offset 10, as the selected index will always cause class majority in the receptive field
# In contrast to offset 10 where equality between classes can be had if selected value on edge
# DONE: Select receptive field based on statistics of the dataset
@cebra.models.register("offset9-model") # --> add that line to register the model!
class MyModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Dropout(0.2),
            nn.Conv1d(num_neurons, 32, 2),
            nn.GELU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, 2),
            nn.GELU(),
            nn.LazyBatchNorm1d(), # I would trust layer or group more (per channel. i.e more per feature of sample instead of over samples)
            #nn.Dropout(0.2),
            cebra_layers._Skip(nn.Conv1d(64, 64, 3),
                               nn.GELU(),nn.LazyBatchNorm1d()),
            #nn.Dropout(0.2),
            cebra_layers._Skip(nn.Conv1d(64, 64, 3),
                               nn.GELU(),nn.LazyBatchNorm1d(),
                               nn.Dropout(0.2)),
            nn.Conv1d(64, num_output, 3),
            #nn.Flatten(),
            #nn.LazyLinear(32*3),
            #nn.GELU(), # num_units*kernel size? (= channels * kernel)
            #nn.LazyBatchNorm1d(),
            #nn.Dropout(0.5),
            #nn.LazyLinear(3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(4, 5)
@cebra.models.register("kernel9-model") # --> add that line to register the model!
class MyModel2(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 9),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 7), nn.GELU(),
            nn.Conv1d(num_units, num_units, 5), nn.GELU(),
            nn.Conv1d(num_units, num_units, 5), nn.GELU(),
            nn.Conv1d(num_units, num_output, 5),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(4+9, 5+9) # such that the full kernel can go on the full mov data >90% of times (90% at least 9 samples)

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

def get_data_channels(sub_test: str, cohort_test: str, df_rmap: pd.DataFrame):
    ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
        "ch"
    ].iloc[0]
    X_test, y_test = get_data_sub_ch(
        ch_all_feat, cohort_test, sub_test, ch_test
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

def plotly_embeddings(X_train_emb,X_test_emb,y_train_discr,y_test,aux=None,type='none',grad=False, nearest=True):
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
    # Also make 1 plot showing the movement gradient (time to next movement onset)
    if grad:
        m0 = np.ones(y_train_discr.shape, dtype=int)
        mask = y_train_discr  # In example True for value you want to give a number
        idx = np.flatnonzero(mask[::-1])  # len 1 iff mask true
        m0[idx[0]] = 0 - idx[0]
        m0[idx[1:]] = idx[:-1] - idx[1:] + 1
        out = np.full(y_train_discr.shape, np.nan, dtype=float)
        out = np.cumsum(m0, axis=0)[::-1]
        out[mask] = 0
        if nearest:
            m02 = np.ones(y_train_discr.shape, dtype=int)
            idx2 = np.flatnonzero(mask)  # len 1 iff mask true
            m02[idx2[0]] = 0 - idx2[0]
            m02[idx2[1:]] = idx2[:-1] - idx2[1:] + 1
            out2 = np.full(y_train_discr.shape, np.nan, dtype=float)
            out2 = np.cumsum(m02, axis=0)
            out2[mask] = 0
            outfull = np.full(y_train_discr.shape, np.nan, dtype=float)
            outfull[:idx2[0]] = out[:idx2[0]]
            outfull[idx2[0]:] = np.min([out[idx2[0]:],out2[idx2[0]:]],axis=0)
            out = outfull
        plotly.offline.plot({'data': [go.Scatter3d(
            x=np.append(X_train_emb[:, 0], X_test_emb[:, 0]),
            y=np.append(X_train_emb[:, 1], X_test_emb[:, 1]),
            z=np.append(X_train_emb[:, 2], X_test_emb[:, 2]),
            mode='markers',
            marker=dict(
                size=2,
                color=np.log(out+1),
                showscale= True,
                colorscale="Viridis_r",
                #cmin=0,
                #cmax=20,
                symbol=np.array(symbollist)[
                    np.array(np.append(np.repeat(0, len(y_train_discr)), np.repeat(1, len(y_test))), dtype=int)],
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

                # Put in TensorDataset with 2d discrete in order cohort - movement
                CohortAuxData = cebra.data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),
                                                         discrete=torch.from_numpy(np.array([coh_aux,y_train],dtype=int).T).type(torch.LongTensor)).to('cuda')
                neural_model = cebra.models.init(
                    name=model_params['model_architecture'],
                    num_neurons=CohortAuxData.input_dimension,
                    num_units=32,
                    num_output=model_params['output_dimension']
                ).to('cuda')

                CohortAuxData.configure_for(neural_model)
                Crit = cebra.models.criterions.LearnableCosineInfoNCE(temperature=model_params['temperature'],
                                                                      min_temperature=model_params[
                                                                          'min_temperature']).to('cuda')

                Opt = torch.optim.Adam(list(neural_model.parameters()) + list(Crit.parameters()), lr=model_params['learning_rate'])
                cebra_model = cebra.solver.init(name="single-session", model=neural_model, criterion=Crit, optimizer=Opt, tqdm_on=True).to(
                    'cuda')
                Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                                  batch_size=model_params['batch_size'], prior=model_params['prior'], cond=model_params['conditional']).to('cuda')
                if model_params['early_stopping']: # TODO: Make the validation set all of the subjects for leave-cohort-out (else early stop on 1 only) OR change to train separate per sub (more accurate)
                    ValidData = cebra.data.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),
                                                             discrete=torch.from_numpy(np.array(y_train,dtype=int).T).type(torch.LongTensor)).to('cuda')
                    ValidData.configure_for(neural_model)
                    Valid_loader = cebra.data.DiscreteDataLoader(dataset=ValidData, num_steps=1,
                                                      batch_size=len(X_test)-len(X_test)%8).to('cuda')
                    cebra_model.fit(loader=Loader,valid_loader=Valid_loader,save_frequency=5,valid_frequency=5,logdir="C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRAsaves")
                    # Save the total losses first before resetting to the early stopping
                    if type(alllosses) == int:
                        alllosses = np.array(np.expand_dims(cebra_model.state_dict()["loss"], axis=0))
                        alltemps = np.array(np.expand_dims(cebra_model.state_dict()["log"]["temperature"], axis=0))
                    else:
                        alllosses = np.concatenate(
                            (alllosses, np.expand_dims(cebra_model.state_dict()["loss"], axis=0)), axis=0)
                        alltemps = np.concatenate(
                            (alltemps, np.expand_dims(cebra_model.state_dict()["log"]["temperature"], axis=0)), axis=0)
                    # Load model with lowest validation loss (INFONCE on validation data)
                    cebra_model.load(logdir="C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRAsaves",filename='checkpoint_best.pth')
                else:
                    cebra_model.fit(loader=Loader)
                    if type(alllosses) == int:
                        alllosses = np.array(np.expand_dims(cebra_model.state_dict()["loss"], axis=0))
                        alltemps = np.array(np.expand_dims(cebra_model.state_dict()["log"]["temperature"], axis=0))
                    else:
                        alllosses = np.concatenate(
                            (alllosses, np.expand_dims(cebra_model.state_dict()["loss"], axis=0)), axis=0)
                        alltemps = np.concatenate(
                            (alltemps, np.expand_dims(cebra_model.state_dict()["log"]["temperature"], axis=0)), axis=0)
                TrainBatches = np.lib.stride_tricks.sliding_window_view(X_train,neural_model.get_offset().__len__(),axis=0)
                X_train_emb = cebra_model.transform(torch.from_numpy(TrainBatches[:]).type(torch.FloatTensor).to('cuda')).to('cuda')
                X_train_emb = X_train_emb.cpu().detach().numpy()

                if model_params['decoder'] == 'KNN':
                    decoder = neighbors.KNeighborsClassifier(
                        n_neighbors=model_params['n_neighbors'], metric=model_params['metric'],
                        n_jobs=model_params['n_jobs'])
                    decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1], dtype=int))
                elif model_params['decoder'] == 'Logistic':
                    decoder = linear_model.LogisticRegression(class_weight='balanced', penalty=None)
                    decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1], dtype=int))
                elif model_params['decoder'] == 'SVM':
                    decoder = SVC(kernel=cosine_similarity)
                    decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1], dtype=int))
                elif model_params['decoder'] == 'FAISS':
                    raise Exception('Not implemented')
                elif model_params['decoder'] == 'MLP':
                    raise Exception('Not implemented')
                elif model_params['decoder'] == 'XGB':
                    decoder = xgboost.sklearn.XGBClassifier()
                    #decoder.set_params(**{'lambda':2})
                    classes_weights = class_weight.compute_sample_weight(
                        class_weight="balanced", y=y_train_discr
                    )
                    decoder.set_params(eval_metric="logloss")
                    decoder.fit(
                        X_train_emb,
                        np.array(y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1], dtype=int),
                        sample_weight=classes_weights,
                    )
                elif model_params['decoder'] == 'KNN_BPP':
                    decoder = kNN_BPP(n_neighbors=model_params['n_neighbors'])
                    decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1], dtype=int))

            # Needed for training once per across cohort
            cohort_prev_it = cohort_test
            # TEST PERMUTATION OF FEATURES
            # rng = np.random.default_rng()
            # X_test_emb = cebra_model.transform(rng.permutation(X_test,axis=1),session_id=0)
            TestBatches = np.lib.stride_tricks.sliding_window_view(X_test,neural_model.get_offset().__len__(),axis=0)
            X_test_emb = cebra_model.transform(torch.from_numpy(TestBatches).type(torch.FloatTensor).to('cuda')).to('cuda')
            X_test_emb = X_test_emb.cpu().detach().numpy()

            ### Embeddings are plotted here
            if show_embedding:
                plotly_embeddings(X_train_emb,X_test_emb,y_train_discr[neural_model.get_offset().left:-neural_model.get_offset().right+1],
                                  y_test[neural_model.get_offset().left:-neural_model.get_offset().right+1],aux=coh_aux,type='coh', grad=False, nearest=False)

            y_test_pr = decoder.predict(X_test_emb)
            ba = metrics.balanced_accuracy_score(y_test[neural_model.get_offset().left:-neural_model.get_offset().right+1], y_test_pr)
            print(ba)
            # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(X_test_emb))

            # cebra.plot_embedding(embedding, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train_cont)
            p_[cohort_test][sub_test] = {}
            p_[cohort_test][sub_test]["performance"] = ba
            #p_[cohort_test][sub_test]["X_test_emb"] = X_test_emb
            p_[cohort_test][sub_test]["y_test"] = y_test[neural_model.get_offset().left:-neural_model.get_offset().right+1]
            p_[cohort_test][sub_test]["y_test_pr"] = y_test_pr
            p_[cohort_test][sub_test]["loss"] = cebra_model.state_dict()["loss"]
            p_[cohort_test][sub_test]["temp"] = cebra_model.state_dict()["log"]["temperature"]

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
            f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA performances/{curtime}_{val_approach}.npy",
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
val_approaches = ["leave_1_cohort_out", "leave_1_sub_out_across_coh"]


for val_approach in val_approaches:
    model_params = {'model_architecture':'offset9-model',
                'batch_size': 512, # Ideally as large as fits on the GPU, min recommended = 512
                'temperature_mode':'auto', # Constant or auto
                'temperature':1,
                'min_temperature':0.1, # If temperature mode = auto this should be set in the desired expected range
                'learning_rate': 0.005, # Set this in accordance to loss function progression in TensorBoard
                'max_iterations': 200,  # 5000, Set this in accordance to the loss functions in TensorBoard
                'time_offsets': 1, # Time offset between samples (Ideally set larger than receptive field according to docs)
                'output_dimension': 3, # Nr of output dimensions of the CEBRA model
                'decoder': 'Logistic', # Choose from "KNN", "Logistic", "SVM", "KNN_BPP"
                'n_neighbors': 35, # KNN & KNN_BPP setting (# of neighbours to consider) 35 works well for 3 output dimensions
                'metric': "euclidean", # KNN setting (For L2 normalized vectors, the ordering of Euclidean and Cosine should be the same)
                'n_jobs': 20, # KNN setting for parallelization
                'all_embeddings':False, # If you want to combine all the embeddings (only when true_msess = True !), currently 1 model is used for the test set --> Make majority
                'true_msess':False, # Make a single model will be made for every subject (Usefull if feature dimension different between subjects)
                'discreteMov':True, # Turn pseudoDiscr to False if you want to test true discrete movement labels (Otherwise this setting will do nothing)
                'pseudoDiscr':False, # Pseudodiscrete meaning direct conversion from int to float
                'gaussSigma':1.5, # Set pseuodDiscr to False for this to take effect and assuming a Gaussian for the movement distribution
                'prior': 'uniform', # Set to empirical or uniform to either sample random or uniform across coh and movement
                'conditional':'mov', # Set to mov or cohmov to either equalize reference-positive in movement or coherenceandmovement
                'early_stopping':False,
                'additional_comment':'PYTORCH_customnegative_mov_EarlyStop1',
                'debug': True} # Debug = True; stops saving the results unnecessarily
    if not model_params['debug']:
        writer = SummaryWriter(log_dir=f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA_logs/{val_approach}/{curtime}")
    t0 = time.time()
    run_CV(val_approach, curtime, model_params,show_embedding=False)
    print(f'runtime: {time.time()-t0}')

