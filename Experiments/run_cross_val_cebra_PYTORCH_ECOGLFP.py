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
import plotly.colors
import plotly.graph_objects as go
import plotly.express as px
import xgboost
from sklearn.utils import class_weight
from Experiments.utils.knn_bpp import kNN_BPP
# from einops.layers.torch import Rearrange # --> Can add to the model to reshape to fit 2dConv maybe
from Experiments.utils.cebracustom import CohortDiscreteDataLoader
import time
from Experiments.utils.ExtraTorchFunc import Attention, WeightAttention, MatrixAttention, AttentionWithContext
torch.backends.cudnn.benchmark = True

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
    os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_noraw.npy"),
    allow_pickle="TRUE",
).item()
df_best_rmap = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch.csv")
df_updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")


# offset9 should be better than offset 10, as the selected index will always cause class majority in the receptive field
# In contrast to offset 10 where equality between classes can be had if selected value on edge
# DONE: Select receptive field based on statistics of the dataset
@cebra.models.register("offset9-model") # --> add that line to register the model!
class MyModel(_OffsetModel, ConvolutionalModelMixin):
# TODO: Add a way to incorporate static and categorical features --> I guess UPDRS as feature (age maybe also) and categorical (gender) through embedding
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
                         )
        self.drop02 = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(num_neurons, 32, 2)
        self.GELU = nn.GELU()
        self.BN = nn.LazyBatchNorm1d() # Can do some work to figure out actual sizes and not use Lazy
        self.BN2 = nn.LazyBatchNorm1d()
        self.BN3 = nn.LazyBatchNorm1d()
        self.BN4 = nn.LazyBatchNorm1d()
        self.conv2 = nn.Conv1d(32, 64, 2)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.conv4 = nn.Conv1d(64, 64, 3)
        self.skipconv3 = cebra_layers._Skip(self.conv3,self.GELU,self.BN3)
        self.skipconv4 = cebra_layers._Skip(self.conv4,self.GELU,self.BN4,self.drop02)
        self.convout = nn.Conv1d(64, num_output, num_output)
        #nn.Flatten(),
        #nn.LazyLinear(32*3),
        #nn.GELU(), # num_units*kernel size? (= channels * kernel)
        #nn.LazyBatchNorm1d(),
        #nn.Dropout(0.5),
        #nn.LazyLinear(3),
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                m.bias.zero_()
    def forward(self,x):
        x = self.drop02(x)
        x = self.BN(self.GELU(self.conv1(x)))
        x = self.drop02(x)
        x = self.BN2(self.GELU(self.conv2(x)))
        # Note that these are cropped skip connections
        x = self.skipconv3(x)
        x = self.skipconv4(x)
        x = self.convout(x)
        return torch.nn.functional.normalize(torch.squeeze(x),p=2,dim=1)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model
    def get_offset(self):
        return cebra.data.Offset(4, 5)


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
        compress = 9
        num_layers = 1
        grudrop = 0
        self.leftset = 5
        self.rightset = 4
        self.rnn = nn.GRU(numfeatures,hidden,num_layers,dropout=grudrop,bidirectional=bidir,batch_first=True)
        self.fc = nn.Linear(hidden*(1+int(bidir)),num_output)

        self.gru_attention1 = Attention(hidden , self.leftset+self.rightset)
        self.gru_attention2 = Attention(hidden, self.leftset + self.rightset)
        self.feat_attention = WeightAttention(self.leftset + self.rightset,numfeatures)
        self.mat_attention = MatrixAttention(self.leftset + self.rightset,numfeatures)
        self.BN = nn.LazyBatchNorm1d()
        self.LN = nn.LayerNorm([numfeatures, compress])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.linfeat = nn.Linear(numfeatures, numfeatures, bias=False)
        # Reduce expand
        self.linred = nn.Linear(numfeatures, compress)
        self.linexp = nn.Linear(compress,numfeatures)
        self.relu = nn.ReLU()
        # Linear to compress time
        self.lintimecomp = nn.Linear(compress,1)
        self.sigmoid = nn.Sigmoid()
        weight = torch.zeros(numfeatures,1)
        nn.init.xavier_uniform_(weight)
        self.learnedscale = nn.Parameter(weight)
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
        #### Feature Attention: # TODO: Time/Feature attention (now compresses time to query point) but prob changes over time
        #### Reduce the time domain first:
        #### Avg pool the features
        #y = self.avg_pool(x)
        #### Take the exact sample (seems to work ok)
        #y = x[:,:,self.leftset].unsqueeze(2)
        #### Max pool
        #y = self.max_pool(x)
        #### FC to compress (let FC find the function to compr over time)
        #y = self.lintimecomp(x)
        #y = self.relu(y) # Optionally with Non-linearity afterwards

        #### Use FC to find corresponding weights, directly or trying to compress :
        #y = self.linfeat(y.transpose(-1, -2)).transpose(-1, -2)
        #### Or with compression inbetween:
        #y = self.linred(y.transpose(-1, -2))
        #y = self.relu(y)
        #y = self.linexp(y).transpose(-1, -2)
        #### Scaling like that for time attention (Find weights per features)
        scale = self.feat_attention(x)
        ### Time-Feature matrix scaling
        #scale = self.mat_attention(x)
        ### Some very naive implementation to scale
        #x = x * self.learnedscale.unsqueeze(0).expand_as(x)
        #### Scale weights between 0 and 1 & Apply
        #scale = self.sigmoid(y) # Save / visualize y here if you want to know channel/feat 'weights'/'scale'
        #x = x * scale.expand_as(x) # Apply the learned Channel / Feature Attention weights
        #x = self.BN(x) # To prevent vanishing grad from [0,1] scaling

        #### Save scaling
        #self.savedscales = self.learnedscale.repeat(1,x.shape[0]).T.unsqueeze(2)
        self.savedscales = scale
        x = x.permute(0,2,1)
        out, hidden = self.rnn(x)
        #### If unidirectional:
        ### Either take the last output (RNN has seen the whole dataset)
        out = self.fc(out[:,-1,:])
        ### Or use attention to take into account more outputs
        #out = self.fc(self.gru_attention1(out))
        #### If Bidirectional
        ### Either take the last output from both the directions
        #out = self.fc(torch.concat([out[:,-1,:2],out[:,0,2:]],1)) # Takes last pred from both sides ????
        ### Or take let attention optimize either together or separate where the forward and backward should be looking
        #out1 = self.gru_attention1(out[:,:,:2])
        #out2 = self.gru_attention2(out[:,:,2:])
        #out = self.fc(torch.concat([out1, out2], 1))
        return torch.nn.functional.normalize(torch.squeeze(out),p=2,dim=1)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model
    def get_offset(self):
        return cebra.data.Offset(self.leftset, self.rightset)
@cebra.models.register("offset9UPDRS-model") # --> add that line to register the model!
class MyModel(_OffsetModel, ConvolutionalModelMixin):
# TODO: Add a way to incorporate static and categorical features --> I guess UPDRS as feature (age maybe also) and categorical (gender) through embedding
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
                         )
        self.drop02 = nn.Dropout(0.2)
        self.drop1d02 = nn.Dropout1d(0.2)
        self.drop04 = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(num_neurons-1, 64, 2)
        self.GELU = nn.GELU()
        self.BN = nn.LazyBatchNorm1d() # Can do some work to figure out actual sizes and not use Lazy
        self.BN2 = nn.LazyBatchNorm1d()
        self.BN3 = nn.LazyBatchNorm1d()
        self.BN4 = nn.LazyBatchNorm1d()
        self.BN5 = nn.LazyBatchNorm1d()
        self.LN = nn.LayerNorm([64,8])
        self.LN2 = nn.LayerNorm([64,7])
        self.LN3 = nn.LayerNorm([64,5])
        self.LN4 = nn.LayerNorm([8,3])
        #self.LN5 = nn.LayerNorm([32, 8])
        self.conv2 = nn.Conv1d(64, 64, 2)
        self.conv3 = nn.Conv1d(64, 64, 3)
        #self.conv4 = nn.Conv1d(64, 64, 3)
        self.conv4alt = nn.Conv1d(64, 8, 3)
        #self.skipconv2 = cebra_layers._Skip(self.conv2, self.GELU, self.LN3)
        self.skipconv3 = cebra_layers._Skip(self.conv3,self.GELU,self.LN3)
        #self.skipconv4 = cebra_layers._Skip(self.conv4,self.GELU,self.LN4)
        #self.convout = nn.Conv1d(32, num_output, 3)
        self.flat = nn.Flatten()
        self.linint = nn.LazyLinear(32)
        self.linout = nn.LazyLinear(num_output)
        #nn.LazyLinear(32*3),
        #nn.GELU(), # num_units*kernel size? (= channels * kernel)
        #nn.LazyBatchNorm1d(),
        #nn.Dropout(0.5),
        #nn.LazyLinear(3),
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                m.bias.zero_()
    def forward(self,x):
        UPDRS = x[:,-1,4] # Take the UPDRS score of the sample corr to the label
        x = x[:,0:-1,:]
        #x = self.drop02(x)
        x = self.BN(self.GELU(self.conv1(x))) # In: 512,36,9, Out: 512,32,8
        #x = self.drop02(x)
        x = self.BN2(self.GELU(self.conv2(x))) # Out: 512,64,7
        # Note that these are cropped skip connections
        x = self.skipconv3(x) # Out: 512,64,5
        x = self.BN4(self.GELU(self.conv4alt(x))) # Out: 512,64,3
        x = self.flat(x) # Out: 512,192
        #x = self.BN5(self.GELU(self.linint(x))) # Out: 512,32
        # Add the UPDRS score as a feature alongside x (now 32) # of other processed
        x = self.GELU(self.linint(torch.concat((x,UPDRS.expand((1,x.size(dim=0))).T),1))) # UPDRS
        x = self.linout(x) # Out: 512,3
        return torch.squeeze(x)

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
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                m.bias.zero_()
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

def get_data_sub_ch(channel_all, cohort, sub, ch, UPDRS=None):

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

    if UPDRS:
        X_train = np.concatenate((X_train,np.expand_dims(np.repeat((UPDRS-41.21052632)
/12.29496665
,len(X_train)),1)),1)

    return X_train, y_train

def get_data_channels(sub_test: str, cohort_test: str, df_rmap: pd.DataFrame, naiveupdrs=False):
    ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
        "ch"
    ].iloc[0]
    if not naiveupdrs:
        X_test, y_test = get_data_sub_ch(
            ch_all_feat, cohort_test, sub_test, ch_test
        )
    else:
        UPDRS = df_updrs.query("cohort == @cohort_test and sub == @sub_test")['UPDRS_total'].iloc[0]
        if np.isnan(UPDRS): # TODO: Properly fix that not all subjects have UPDRS
            UPDRS = 41.21052632 # QUICK PATCHUP TO RUN A TEST
        X_test, y_test = get_data_sub_ch(
            ch_all_feat, cohort_test, sub_test, ch_test, UPDRS
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
    if save:
        writer.add_figure('Performance_Figure', plt.gcf(), 0)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

def plotly_embeddings(X_train_emb,X_test_emb,y_train_discr,y_test,test_coh,aux=None,type='none',grad=False, nearest=True,model=None,val=None):
#    Plot together with the test embedding (For coh auxillary)
    symbollist = ['circle', 'diamond', 'square','cross', 'x']

    if type == 'coh':
        colorlist = px.colors.qualitative.D3
        import plotly.io as pio
        pio.renderers.default = 'browser'
        # Decompose data per legendclass (i.e. unique color-symbol)
        colorlabels = np.array(np.append(y_train_discr, np.array(y_test, dtype=int) + 2), dtype=int)
        symbollabels = np.array(np.append(aux, np.repeat(max(aux) + 1, len(y_test))), dtype=int)
        uniquelabels = np.zeros_like(colorlabels)
        viridis_colors, _ = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors)
        colorlist = [get_continuous_color(colorscale, intermed=0.25),get_continuous_color(colorscale, intermed=0.75),get_continuous_color(colorscale, intermed=0.0),get_continuous_color(colorscale, intermed=1)]

        # Label manipulation to split properly
        # Train data
        uniquelabels[symbollabels != np.max(symbollabels)] = colorlabels[symbollabels != np.max(symbollabels)] + 2*symbollabels[symbollabels != np.max(symbollabels)]
        # Test data
        uniquelabels[symbollabels == np.max(symbollabels)] = (colorlabels[symbollabels == np.max(symbollabels)] - 2) + 2*symbollabels[symbollabels == np.max(symbollabels)]
        colororder = np.append(np.tile([0,1],np.max(symbollabels)),[2,3])
        symbolorder = np.repeat(np.arange(np.max(symbollabels)+1),2)
        alldata = np.append(X_train_emb, X_test_emb,axis=0)
        names = []
        if val == 'leave_1_cohort_out':
            traincoh = cohorts.copy()
            traincoh.remove(test_coh)
            for i in range(np.max(symbollabels)):
                names.append(f'Train rest samples, {traincoh[i]}')
                names.append(f'Train movement samples, {traincoh[i]}') # add proper cohort name
            names.append(f'Test rest samples, {test_coh}')
            names.append(f'Test movement samples,{test_coh}')
        elif val == "leave_1_sub_out_within_coh":
            names.append(f'Train rest samples, {test_coh}')
            names.append(f'Train movement samples,{test_coh}')
            names.append(f'Test rest samples, {test_coh}')
            names.append(f'Test movement samples,{test_coh}')
        elif val == "leave_1_sub_out_across_coh":
            traincoh = cohorts.copy()
            for i in range(np.max(symbollabels)):
                names.append(f'Train rest samples, {traincoh[i]}')
                names.append(f'Train movement samples, {traincoh[i]}') # add proper cohort name
            names.append(f'Test rest samples, {test_coh}')
            names.append(f'Test movement samples,{test_coh}')
        data = []
        fig = go.Figure()
        for i in range(np.max(uniquelabels) + 1):
            fig.add_trace(go.Scatter3d(
                x=alldata[uniquelabels == i, 0],
                y=alldata[uniquelabels == i, 1],
                z=alldata[uniquelabels == i, 2],
                name=names[i],
                mode='markers',
                showlegend=True,
                marker=dict(
                    size=2,
                    color=np.array(colorlist)[colororder[i]],
                    symbol=np.array(symbollist)[symbolorder[i]],
                    # set color to an array/list of desired values
                    opacity=0.8)))
        fig.update_layout(template='simple_white',
            scene=dict(
                xaxis=dict(visible=False, backgroundcolor="rgba(0, 0, 0,0)", gridcolor="gray"),
                yaxis=dict(visible=False, backgroundcolor="rgba(0, 0, 0,0)", gridcolor="gray"),
                zaxis=dict(visible=False, backgroundcolor="rgba(0, 0, 0,0)", gridcolor="gray"),
                bgcolor='white',
            ),
            legend=dict(
                font=dict(family="Arial", size=10),
                yanchor="middle",
                y=0.5,
                itemsizing='constant'
            ),
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        config = {
            'toImageButtonOptions': {
                'format': 'svg',  # one of png, svg, jpeg, webp
                'filename': 'custom_image',
                'height': 1000,
                'width': 1400,
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        if model:
            z = lambda x, y: (-model.intercept_[0] - model.coef_[0][0] * x - model.coef_[0][1] * y) / model.coef_[0][2]
            tmp = np.linspace(-1.1, 1.1, 20)
            x, y = np.meshgrid(tmp, tmp)
            fig.add_traces(go.Surface(x=x, y=y, z=z(x,y), name='pred_surface',showlegend=False))
        fig.show(config=config)

        # plotly.offline.plot({'data': [go.Scatter3d(
        #     x=np.append(X_train_emb[:, 0],X_test_emb[:, 0]),
        #     y=np.append(X_train_emb[:, 1],X_test_emb[:, 1]),
        #     z=np.append(X_train_emb[:, 2],X_test_emb[:, 2]),
        #     mode='markers',
        #     scene=dict(
        #         xaxis=dict(visible=False),
        #         yaxis=dict(visible=False),
        #         zaxis=dict(visible=False)
        #     ),
        #     marker=dict(
        #         size=2,
        #         color=np.array(colorlist)[np.array(np.append(y_train_discr,np.array(y_test,dtype=int)+2), dtype=int)],
        #         symbol=np.array(symbollist)[np.array(np.append(aux,np.repeat(max(aux)+1,len(y_test))), dtype=int)],
        #         # set color to an array/list of desired values
        #         opacity=0.8))]}, auto_open=True)

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

def run_CV(val_approach,curtime,model_params,show_embedding=False,embeddingconsistency=False,showfeatureweights=False,Testphase=False):
    train_select = ch_all_train
    test_select = ch_all_test
    alldat = ch_all_feat
    p_ = {}
    batotal = []
    bacohortlist = []
    cohort_prev_it = ""
    alllosses = 0
    alltemps = 0
    trainembeddings = [] # list of embeddings
    trainID = []
    trainembeddinglabels = [] # list of corr labels (mov/nomov)
    testembeddings = [] # list of embeddings
    testID = []
    testembeddinglabels = [] # list of corr labels (mov/nomov)
    for cohort_test in cohorts:
        if cohort_test not in p_:
            p_[cohort_test] = {}
        bacohort = []
        if Testphase:
            subtests = test_select[cohort_test].keys()
        else:
            subtests = train_select[cohort_test].keys()
        for sub_test in subtests: # test_select for unseen test data
            print('Val approach, cohort, subject:', val_approach, cohort_test, sub_test)
            if sub_test not in p_[cohort_test]:
                p_[cohort_test][sub_test] = {}
            X_test, y_test = get_data_channels(
                sub_test, cohort_test, df_rmap=df_best_rmap
            )

            # if statement to keep the same model for all subjects of leave 1 out cohort
            if (val_approach == "leave_1_cohort_out" and cohort_test != cohort_prev_it) or val_approach != "leave_1_cohort_out":
                cohorts_train = get_patients_train_dict(
                    sub_test, cohort_test, val_approach=val_approach, data_select=ch_all # Use all data in training (except test data, which gets rejected in the function)
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

                Opt = torch.optim.Adam(list(neural_model.parameters()) + list(Crit.parameters()), lr=model_params['learning_rate'],weight_decay=0)
                cebra_model = cebra.solver.init(name="single-session", model=neural_model, criterion=Crit, optimizer=Opt, tqdm_on=True).to(
                    'cuda')
                Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                                  batch_size=model_params['batch_size'], prior=model_params['prior'], cond=model_params['conditional']).to('cuda')
                if model_params['early_stopping']: # TODO: Make the validation set all of the subjects for leave-cohort-out (else early stop on 1 only) OR change to train separate per sub (more accurate)
                    ValidData = cebra.data.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),
                                                             discrete=torch.from_numpy(np.array([np.repeat(0,len(y_test)),y_test],dtype=int).T).type(torch.LongTensor)).to('cuda')
                    ValidData.configure_for(neural_model)
                    Valid_loader = CohortDiscreteDataLoader(dataset=ValidData, num_steps=1,
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
                if showfeatureweights:
                    scales = cebra_model.model.savedscales.detach().cpu().numpy()
                    if scales.shape[2] > 1:
                        somechannel = list(alldat[cohort_test][sub_test].keys())[0]
                        somerun = list(alldat[cohort_test][sub_test][somechannel].keys())[0]
                        features = np.array(list(alldat[cohort_test][sub_test][somechannel][somerun]['feature_names']))
                        meanscale = np.mean(scales,0)
                        g = sns.heatmap(meanscale,cmap="viridis")
                        g.set_title("Learned Time - Feature scaling")
                        g.set_yticks(np.arange(len(features))+0.5)
                        g.set_yticklabels(features, rotation=0)
                        for tick_label in g.axes.get_xticklabels():
                            if tick_label.get_text() == '4':
                                tick_label.set_color("red")
                        plt.show()

                    else:
                        scales = scales[:,:,0]
                        somechannel = list(alldat[cohort_test][sub_test].keys())[0]
                        somerun = list(alldat[cohort_test][sub_test][somechannel].keys())[0]
                        features = np.array(list(alldat[cohort_test][sub_test][somechannel][somerun]['feature_names']))
                        med = np.median(scales,0)
                        sorted_desc = np.argsort(med)[::-1]
                        g = sns.boxplot(scales[:,sorted_desc])
                        g.set_title("Scaling applied by the feature attention module [0,1]")
                        g.set_xticklabels(features[sorted_desc], rotation=45, ha='right')
                        plt.show()
                rightbound = -neural_model.get_offset().right+1
                tillend=False
                if rightbound == 0:
                    tillend = True
                if embeddingconsistency: # TODO: Fix computation of embedding consistencies
                    trainembeddings.append(X_train_emb)
                    trainID.append('+'.join([coh for coh in cohorts if coh != cohort_test]))
                    if tillend:
                        trainembeddinglabels.append(
                            y_train_discr[neural_model.get_offset().left:])
                    else:
                        trainembeddinglabels.append(y_train_discr[neural_model.get_offset().left:rightbound])

                if model_params['decoder'] == 'KNN':
                    decoder = neighbors.KNeighborsClassifier(
                        n_neighbors=model_params['n_neighbors'], metric=model_params['metric'],
                        n_jobs=model_params['n_jobs'])
                    if tillend:
                        decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:], dtype=int))
                    else:
                        decoder.fit(X_train_emb, np.array(
                            y_train_discr[neural_model.get_offset().left:rightbound],
                            dtype=int))
                elif model_params['decoder'] == 'Logistic':
                    decoder = linear_model.LogisticRegression(class_weight='balanced', penalty=None)
                    if tillend:
                        decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:], dtype=int))
                    else:
                        decoder.fit(X_train_emb, np.array(
                            y_train_discr[neural_model.get_offset().left:rightbound],
                            dtype=int))
                elif model_params['decoder'] == 'SVM':
                    decoder = SVC(kernel=cosine_similarity)
                    if tillend:
                        decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:], dtype=int))
                    else:
                        decoder.fit(X_train_emb, np.array(
                            y_train_discr[neural_model.get_offset().left:rightbound],
                            dtype=int))
                elif model_params['decoder'] == 'FAISS':
                    raise Exception('Not implemented')
                elif model_params['decoder'] == 'MLP':
                    raise Exception('Not implemented')
                elif model_params['decoder'] == 'XGB':
                    decoder = xgboost.sklearn.XGBClassifier()
                    #decoder.set_params(**{'lambda':2})
                    if tillend:
                        classes_weights = class_weight.compute_sample_weight(
                            class_weight="balanced", y=y_train_discr[neural_model.get_offset().left:]
                        )
                        decoder.set_params(eval_metric="auc")
                        decoder.fit(
                            X_train_emb,
                            np.array(y_train_discr[neural_model.get_offset().left:], dtype=int),
                            sample_weight=classes_weights,
                        )
                    else:
                        classes_weights = class_weight.compute_sample_weight(
                            class_weight="balanced", y=y_train_discr[neural_model.get_offset().left:rightbound]
                        )
                        decoder.set_params(eval_metric="auc")
                        decoder.fit(
                            X_train_emb,
                            np.array(y_train_discr[neural_model.get_offset().left:rightbound], dtype=int),
                            sample_weight=classes_weights,
                        )
                elif model_params['decoder'] == 'KNN_BPP':
                    decoder = kNN_BPP(n_neighbors=model_params['n_neighbors'])
                    if tillend:
                        decoder.fit(X_train_emb, np.array(y_train_discr[neural_model.get_offset().left:], dtype=int))
                    else:
                        decoder.fit(X_train_emb, np.array(
                            y_train_discr[neural_model.get_offset().left:rightbound],
                            dtype=int))

            # Needed for training once per across cohort
            cohort_prev_it = cohort_test
            # TEST PERMUTATION OF FEATURES
            # rng = np.random.default_rng()
            # X_test_emb = cebra_model.transform(rng.permutation(X_test,axis=1),session_id=0)
            TestBatches = np.lib.stride_tricks.sliding_window_view(X_test,neural_model.get_offset().__len__(),axis=0)
            X_test_emb = cebra_model.transform(torch.from_numpy(TestBatches).type(torch.FloatTensor).to('cuda')).to('cuda')
            X_test_emb = X_test_emb.cpu().detach().numpy()
            if embeddingconsistency:
                testembeddings.append(X_test_emb)
                testID.append(sub_test)
                if tillend:
                    testembeddinglabels.append(y_test[neural_model.get_offset().left:])
                else:
                    testembeddinglabels.append(
                        y_test[neural_model.get_offset().left:rightbound])
            ### Embeddings are plotted here
            if show_embedding:
                if tillend:
                    plotly_embeddings(X_train_emb,X_test_emb,y_train_discr[neural_model.get_offset().left:],
                                  y_test[neural_model.get_offset().left:],cohort_test,aux=coh_aux[neural_model.get_offset().left:],type='coh', grad=False, nearest=False,model=None,val=val_approach)
                else:
                    plotly_embeddings(X_train_emb, X_test_emb, y_train_discr[
                                                               neural_model.get_offset().left:rightbound],
                                      y_test[neural_model.get_offset().left:rightbound],cohort_test,
                                      aux=coh_aux[neural_model.get_offset().left:rightbound], type='coh', grad=False, nearest=False,model=None,val=val_approach)

            y_test_pr = decoder.predict(X_test_emb)
            if tillend:
                ba = metrics.balanced_accuracy_score(y_test[neural_model.get_offset().left:], y_test_pr)
            else:
                ba = metrics.balanced_accuracy_score(
                    y_test[neural_model.get_offset().left:rightbound], y_test_pr)
            print(ba)
            # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(X_test_emb))

            # cebra.plot_embedding(embedding, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train_cont)
            p_[cohort_test][sub_test] = {}
            p_[cohort_test][sub_test]["performance"] = ba
            #p_[cohort_test][sub_test]["X_test_emb"] = X_test_emb
            if tillend:
                p_[cohort_test][sub_test]["y_test"] = y_test[neural_model.get_offset().left:]
            else:
                p_[cohort_test][sub_test]["y_test"] = y_test[
                                                      neural_model.get_offset().left:rightbound]
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
        p_[cohort_test]['mean_ba'] = np.mean(bacohort)
        p_[cohort_test]['std_ba'] = np.std(bacohort)

        # Compute for each of the training runs the consistency between the test subject embeddings
        if embeddingconsistency: # For each of the training
            # Maybe use iterative closest point analysis, or see how the dataset method can be used
            # The dataset method bins based on label and then matches the mean of each label, This is not nice with
            # only 2 means (undefined)
            scores_sub, pairs_sub, datasets_sub = cebra.sklearn.metrics.consistency_score(embeddings=testembeddings,
                                                                                             #dataset_ids=testID,
                                                                                             between="runs")
            plt.figure()
            cebra.plot_consistency(scores_sub, pairs_sub, datasets_sub, vmin=0, vmax=100,
                                   title="Between-subject consistencies")
            plt.show()
    # Compute the consistency between training embeddings
    if embeddingconsistency:
        scores_coh, pairs_coh, datasets_coh = cebra.sklearn.metrics.consistency_score(embeddings=trainembeddings,
                                                                                         #dataset_ids=trainID,
                                                                                         between="runs")
        fig = plt.figure(figsize=(10, 4))
        cebra.plot_consistency(scores_coh, pairs_coh, datasets_coh, vmin=0, vmax=100,
                               title="Between-cohort consistencies")
    # Save the output if it was not a debug run
    if not model_params['debug']:
        # After the run
        p_['total_mean_ba'] = mean_ba
        p_['total_std'] = np.std(batotal)
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
perflist = []
val_approaches = ["leave_1_sub_out_within_coh","leave_1_cohort_out","leave_1_sub_out_across_coh"]
cohorts = [ "Pittsburgh","Beijing", "Berlin"]#, "Washington"]

for val_approach in val_approaches:
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
                'additional_comment':'TEST_BidirGRUModel_FeatAttention',
                'debug': True} # Debug = True; stops saving the results unnecessarily
    if not model_params['debug']:
        writer = SummaryWriter(log_dir=f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA_logs/{val_approach}/{curtime}")

    t0 = time.time()
    run_CV(val_approach, curtime, model_params,show_embedding=True, embeddingconsistency=False,showfeatureweights=False,Testphase=False)
    print(f'runtime: {time.time()-t0}')

