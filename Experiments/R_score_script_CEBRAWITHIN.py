from sklearn.model_selection import KFold, cross_validate, cross_val_score
import numpy as np
import os
import pandas as pd
from torch import nn
import torch
import cebra
import cebra.models
import cebra.data
import cebra.distributions
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin, cebra_layers
from sklearn import metrics, neighbors
from sklearn import linear_model
from Experiments.utils.cebracustom import CohortDiscreteDataLoader
from Experiments.utils.ExtraTorchFunc import Attention, WeightAttention, MatrixAttention, AttentionWithContext
torch.backends.cudnn.benchmark = True

# import the data
ch_all = np.load(
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "Cleaned_channel_all.npy"),
    allow_pickle="TRUE",
).item()

df_perf = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_all_features.csv")

@cebra.models.register("offset9RNN-model") # --> add that line to register the model!
class MyModel(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
                         )
        self.savedscales = None
        self.savedparam = None
        bidir = False
        hidden = 2
        numfeatures = 36
        num_layers = 1
        grudrop = 0
        self.leftset = 5
        self.rightset = 4
        self.rnn = nn.GRU(numfeatures,hidden,num_layers,dropout=grudrop,bidirectional=bidir,batch_first=True)
        self.fc = nn.Linear(hidden*(1+int(bidir)),num_output)

        self.feat_attention = WeightAttention(self.leftset + self.rightset,numfeatures)
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)
    def forward(self,x):
        scale = self.feat_attention(x)
        self.savedscales = scale
        x = x.permute(0,2,1)
        out, hidden = self.rnn(x)
        out = self.fc(out[:,-1,:])
        return torch.nn.functional.normalize(torch.squeeze(out),p=2,dim=1)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model
    def get_offset(self):
        return cebra.data.Offset(self.leftset, self.rightset)

model_params = {'model_architecture':'offset9RNN-model',
                'batch_size': 512, # Ideally as large as fits on the GPU, min recommended = 512
                'temperature_mode':'auto', # Constant or auto
                'temperature':1,
                'min_temperature':0.1, # If temperature mode = auto this should be set in the desired expected range
                'learning_rate': 0.005, # Set this in accordance to loss function progression in TensorBoard
                'max_iterations': 500,  # 5000, Set this in accordance to the loss functions in TensorBoard
                'time_offsets': 1, # Time offset between samples (Ideally set larger than receptive field according to docs)
                'output_dimension': 3, # Nr of output dimensions of the CEBRA model
                'decoder': 'Logistic', # Choose from "KNN", "Logistic", "SVM", "KNN_BPP or XGB"
                'n_neighbors': 35, # KNN & KNN_BPP setting (# of neighbours to consider) 35 works well for 3 output dimensions
                'metric': "euclidean", # KNN setting (For L2 normalized vectors, the ordering of Euclidean and Cosine should be the same)
                'n_jobs': -1, # KNN setting for parallelization
                'all_embeddings':False, # If you want to combine all the embeddings (only when true_msess = True !), currently 1 model is used for the test set --> Make majority
                'true_msess':False, # Make a single model will be made for every subject (Usefull if feature dimension different between subjects)
                'discreteMov':True, # Turn pseudoDiscr to False if you want to test true discrete movement labels (Otherwise this setting will do nothing)
                'pseudoDiscr':False, # Pseudodiscrete meaning direct conversion from int to float
                'gaussSigma':1.5, # Set pseuodDiscr to False for this to take effect and assuming a Gaussian for the movement distribution
                'prior': 'uniform', # Set to empirical or uniform to either sample random or uniform across coh and movement
                'conditional':'mov', # Set to mov or cohmov to either equalize reference-positive in movement or coherenceandmovement
                'early_stopping':False,
                'additional_comment':'BidirGRUModel_FeatAttention',
                'debug': True} # Debug = True; stops saving the results unnecessarily
def RunCebra(X_train,coh_aux,y_train,X_test,y_test):
    CohortAuxData = cebra.data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),
                                             discrete=torch.from_numpy(np.array([coh_aux, y_train], dtype=int).T).type(
                                                 torch.LongTensor)).to('cuda')
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

    Opt = torch.optim.Adam(list(neural_model.parameters()) + list(Crit.parameters()), lr=model_params['learning_rate'],
                           weight_decay=0)
    cebra_model = cebra.solver.init(name="single-session", model=neural_model, criterion=Crit, optimizer=Opt,
                                    tqdm_on=True).to(
        'cuda')
    Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                      batch_size=model_params['batch_size'], prior=model_params['prior'],
                                      cond=model_params['conditional']).to('cuda')
    cebra_model.fit(loader=Loader)
    TrainBatches = np.lib.stride_tricks.sliding_window_view(X_train, neural_model.get_offset().__len__(), axis=0)
    X_train_emb = cebra_model.transform(torch.from_numpy(TrainBatches[:]).type(torch.FloatTensor).to('cuda')).to('cuda')
    X_train_emb = X_train_emb.cpu().detach().numpy()
    rightbound = -neural_model.get_offset().right + 1
    tillend = False
    if rightbound == 0:
        tillend = True
    decoder = linear_model.LogisticRegression(class_weight='balanced', penalty=None)
    if tillend:
        decoder.fit(X_train_emb, np.array(y_train[neural_model.get_offset().left:], dtype=int))
    else:
        decoder.fit(X_train_emb, np.array(
            y_train[neural_model.get_offset().left:rightbound],
            dtype=int))
    TestBatches = np.lib.stride_tricks.sliding_window_view(X_test, neural_model.get_offset().__len__(), axis=0)
    X_test_emb = cebra_model.transform(torch.from_numpy(TestBatches).type(torch.FloatTensor).to('cuda')).to('cuda')
    X_test_emb = X_test_emb.cpu().detach().numpy()
    y_test_pr = decoder.predict(X_test_emb)
    if tillend:
        ba = metrics.balanced_accuracy_score(y_test[neural_model.get_offset().left:], y_test_pr)
    else:
        ba = metrics.balanced_accuracy_score(
            y_test[neural_model.get_offset().left:rightbound], y_test_pr)
    return ba

# set features to use (do fft separately, as not every has to be computed)
features = ['Hjorth', 'Sharpwave', 'fooof', 'bursts','fft', 'combined']
performances = []
featuredim = ch_all['Berlin']['002']['ECOG_L_1_SMC_AT-avgref']['sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
idxlist = []
for i in range(len(features)-1):
    idx_i = np.nonzero(np.char.find(featuredim, features[i])+1)[0]
    idxlist.append(idx_i)
# Add the combined feature idx list (and one for special case subject 001 Berlin
idxlist_Berlin_001 = idxlist.copy()
idxlist_Berlin_001[3] = np.add(idxlist_Berlin_001[3],1)
idxlist.append(np.concatenate(idxlist))
idxlist_Berlin_001.append(np.concatenate(idxlist_Berlin_001))

kf = KFold(n_splits = 3, shuffle = False)
model = linear_model.LogisticRegression(class_weight="balanced")
bascorer = metrics.make_scorer(metrics.balanced_accuracy_score)
performancedict = {}
meanlist = []
features = ['combined']
idxofmax = np.sort(list(df_perf.groupby(['cohort','sub'])['ba_combined'].idxmax()))
for ch_i in range(len(idxofmax)):
    idx = idxofmax[ch_i]
    cohort = df_perf.iloc[idx]['cohort']
    sub = df_perf.iloc[idx]['sub']
    chname = df_perf.iloc[idx]['ch']
    if not cohort in performancedict:
        performancedict[cohort] = {}
    performancedict[cohort][sub] = {}
    performancedict[cohort][sub][chname] = {}
    performancedict[cohort][sub][chname]['ba'] = {}
    performancedict[cohort][sub][chname]['95%CI'] = {}
    x_concat = []
    y_concat = []
    for runs in ch_all[cohort][sub][chname].keys():
        x_concat.append(
            np.squeeze(ch_all[cohort][sub][chname][runs]['data']))  # Get best ECOG data
        y_concat.append(ch_all[cohort][sub][chname][runs]['label'])
    x_concat = np.concatenate(x_concat, axis=0)
    y_concat = np.concatenate(y_concat, axis=0)
    crossvals = kf.split(x_concat,y_concat)
    scores = []
    for i, (train_index, test_index) in enumerate(crossvals):
        scores.append(RunCebra(x_concat[train_index,:],np.zeros_like(x_concat[train_index,0]),y_concat[train_index],x_concat[test_index,:],y_concat[test_index]))
    performancedict[cohort][sub][chname]['ba'][features[0]] = np.mean(scores)
    print(np.mean(scores))
    meanlist.append(np.mean(scores))
    print(f'Running mean: {np.mean(meanlist)}')
    performancedict[cohort][sub][chname]['95%CI'][features[0]] = np.std(scores) * 2
    performancedict[cohort][sub][chname]['explength'] = len(y_concat)
    performancedict[cohort][sub][chname]['movsamples'] = np.sum(y_concat)

np.save(r'C:\Users\ICN_GPU\Documents\Glenn_Data\WithinChannelCEBRA.npy', performancedict)
