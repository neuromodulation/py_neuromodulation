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
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
torch.backends.cudnn.benchmark = True

import skopt
from skopt.space import Real
from skopt.space import Categorical
from skopt.space import Integer
from skopt import gp_minimize
from skopt.utils import use_named_args

class run_cross_val_cebra:
    '''A class that consists of all functions needed to perform three different kind of cross validations on movement decoding
    data. Features can be set by calling the class with a dictionary containing settings, the desired validation approaches
    and cohorts to run the tests for.'''
    def __init__(self,settings,val_approaches,cohorts,):
        # Statics / Data
        self.ch_all_train = np.load(
            os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "train_channel_all_fft.npy"),
            allow_pickle="TRUE",
        ).item()
        self.ch_all_test = np.load(
            os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "test_channel_all_fft.npy"),
            allow_pickle="TRUE",
        ).item()
        self.ch_all = np.load(
            os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "channel_all_fft.npy"),
            allow_pickle="TRUE",
        ).item()
        self.ch_all_feat = np.load(
            os.path.join(r"C:\Users\ICN_GPU\Documents\Glenn_Data", "TempCleaned2_channel_all.npy"),
            allow_pickle="TRUE",
        ).item()
        self.df_best_rmap = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch_adapted.csv")
        self.df_best_rmap_og = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_best_func_rmap_ch.csv")
        self.df_updrs = pd.read_csv(r"C:\Users\ICN_GPU\Documents\Glenn_Data\df_updrs.csv")
        self.show_embedding = False
        self.embeddingconsistency = False
        self.showfeatureweights = False
        self.Testphase = False
        self.Captum = False
        # Standard val_approach for run CV
        self.val_approach = ''
        self.writer = None

        # Dynamics / Settings and run parameters
        self.model_params = settings
        self.val_approaches = val_approaches
        self.cohorts = cohorts

        # Empty values changing during the run
        self.offset = None
        self.tillend = False

        # Derivatives of the dataset
        self.featuredict, self.idxlist, self.featurelist = self._create_featuredict_and_selected_idx()

    def _create_featuredict_and_selected_idx(self):
        '''Internal function that derives a dictionary containing the features and their index locations from
        the dataset and gather the desired indices to be used in the whole run'''
        features = ['Hjorth', 'fft', 'Sharpwave', 'fooof', 'bursts']
        featuredim = self.ch_all_feat['Berlin']['002']['ECOG_L_1_SMC_AT-avgref'][
            'sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg']['feature_names']
        featuredict = {}
        for i in range(len(features)):
            idx_i = np.nonzero(np.char.find(featuredim, features[i]) + 1)[0]
            featuredict[features[i]] = idx_i
        if self.model_params['features']:
            toselect = self.model_params['features'].split(',')
            idxlist = []
            for featsel in toselect:
                idxlist.append(featuredict[featsel])
            idxlist = np.concatenate(idxlist)
            selfeatures = np.array(featuredim)[idxlist]
        else:
            idxlist = list(range(featuredim)) # use all features if nothing is supplied
            selfeatures = np.array(featuredim)[idxlist]

        return featuredict, idxlist, selfeatures

    def _register_model(self, architecture = ''):
        '''Internal functiofn that is used to register the correct model if the model is a custom one (else this function will do nothing)'''
        if not architecture:
            architecture = self.model_params['model_architecture']
        if not architecture in cebra.models.get_options():
            if architecture == "offset9-model":
                @cebra.models.register("offset9-model")  # --> add that line to register the model!
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
                        self.BN = nn.LazyBatchNorm1d()  # Can do some work to figure out actual sizes and not use Lazy
                        self.BN2 = nn.LazyBatchNorm1d()
                        self.BN3 = nn.LazyBatchNorm1d()
                        self.BN4 = nn.LazyBatchNorm1d()
                        self.conv2 = nn.Conv1d(32, 64, 2)
                        self.conv3 = nn.Conv1d(64, 64, 3)
                        self.conv4 = nn.Conv1d(64, 64, 3)
                        self.skipconv3 = cebra_layers._Skip(self.conv3, self.GELU, self.BN3)
                        self.skipconv4 = cebra_layers._Skip(self.conv4, self.GELU, self.BN4, self.drop02)
                        self.convout = nn.Conv1d(64, num_output, num_output)
                        # nn.Flatten(),
                        # nn.LazyLinear(32*3),
                        # nn.GELU(), # num_units*kernel size? (= channels * kernel)
                        # nn.LazyBatchNorm1d(),
                        # nn.Dropout(0.5),
                        # nn.LazyLinear(3),

                    def init_weights(m):
                        if isinstance(m, nn.Conv2d):
                            torch.nn.init.xavier_uniform_(m.weight)
                            with torch.no_grad():
                                m.bias.zero_()

                    def forward(self, x):
                        x = self.drop02(x)
                        x = self.BN(self.GELU(self.conv1(x)))
                        x = self.drop02(x)
                        x = self.BN2(self.GELU(self.conv2(x)))
                        # Note that these are cropped skip connections
                        x = self.skipconv3(x)
                        x = self.skipconv4(x)
                        x = self.convout(x)
                        return torch.nn.functional.normalize(torch.squeeze(x), p=2, dim=1)

                    # ... and you can also redefine the forward method,
                    # as you would for a typical pytorch model
                    def get_offset(self):
                        return cebra.data.Offset(4, 5)
            elif architecture == "offset9RNN-model":
                @cebra.models.register("offset9RNN-model")  # --> add that line to register the model!
                class MyModel(_OffsetModel, ConvolutionalModelMixin):
                    def __init__(self, num_neurons, num_units, num_output, latent, numlayers, left_set, normalize=True):
                        super().__init__(num_input=num_neurons,
                                         num_output=num_output,
                                         normalize=normalize,
                                         latent = latent,
                                         numlayers = numlayers,
                                         left_set = left_set
                                         )
                        self.savedscales = None
                        self.savedparam = None
                        bidir = False
                        hidden = latent
                        compress = 9
                        num_layers = numlayers
                        grudrop = 0
                        self.leftset = left_set
                        self.rightset = 1
                        self.rnn = nn.GRU(num_neurons, hidden, num_layers, dropout=grudrop, bidirectional=bidir,
                                          batch_first=True)
                        self.fc = nn.Linear(hidden * (1 + int(bidir)), num_output)

                        self.gru_attention1 = Attention(hidden, self.leftset + self.rightset)
                        self.gru_attention2 = Attention(hidden, self.leftset + self.rightset)
                        self.contextattention = AttentionWithContext(hidden, self.leftset + self.rightset)
                        self.feat_attention = WeightAttention(self.leftset + self.rightset, num_neurons)
                        self.mat_attention = MatrixAttention(self.leftset + self.rightset, num_neurons)
                        self.BN = nn.LazyBatchNorm1d()
                        self.LN = nn.LayerNorm([num_neurons, compress])
                        self.avg_pool = nn.AdaptiveAvgPool1d(1)
                        self.max_pool = nn.AdaptiveMaxPool1d(1)
                        self.linfeat = nn.Linear(num_neurons, num_neurons, bias=False)
                        # Reduce expand
                        self.linred = nn.Linear(num_neurons, compress)
                        self.linexp = nn.Linear(compress, num_neurons)
                        self.relu = nn.ReLU()
                        # Linear to compress time
                        self.lintimecomp = nn.Linear(compress, 1)
                        self.sigmoid = nn.Sigmoid()
                        weight = torch.zeros(num_neurons, 1)
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

                    def forward(self, x):
                        #### Feature Attention: # TODO: Time/Feature attention (now compresses time to query point) but prob changes over time
                        #### Reduce the time domain first:
                        #### Avg pool the features
                        # y = self.avg_pool(x)
                        #### Take the exact sample (seems to work ok)
                        # y = x[:,:,self.leftset].unsqueeze(2)
                        #### Max pool
                        # y = self.max_pool(x)
                        #### FC to compress (let FC find the function to compr over time)
                        # y = self.lintimecomp(x)
                        # y = self.relu(y) # Optionally with Non-linearity afterwards

                        #### Use FC to find corresponding weights, directly or trying to compress :
                        # y = self.linfeat(y.transpose(-1, -2)).transpose(-1, -2)
                        #### Or with compression inbetween:
                        # y = self.linred(y.transpose(-1, -2))
                        # y = self.relu(y)
                        # y = self.linexp(y).transpose(-1, -2)
                        #### Scaling like that for time attention (Find weights per features)
                        # scale = self.feat_attention(x)
                        ### Time-Feature matrix scaling
                        # scale = self.mat_attention(x)
                        ### Some very naive implementation to scale
                        # x = x * self.learnedscale.unsqueeze(0).expand_as(x)
                        #### Scale weights between 0 and 1 & Apply
                        # scale = self.sigmoid(y) # Save / visualize y here if you want to know channel/feat 'weights'/'scale'
                        # x = x * scale.expand_as(x) # Apply the learned Channel / Feature Attention weights
                        # x = self.BN(x) # To prevent vanishing grad from [0,1] scaling

                        #### Save scaling
                        # self.savedscales = self.learnedscale.repeat(1,x.shape[0]).T.unsqueeze(2)
                        # self.savedscales = scale
                        x = x.permute(0, 2, 1)
                        out, hidden = self.rnn(x)
                        #### If unidirectional:
                        ### Either take the last output (RNN has seen the whole dataset)
                        out = self.fc(out[:, -1, :])
                        ### Or use attention to take into account more outputs
                        #out = self.fc(self.contextattention(out))
                        #### If Bidirectional
                        ### Either take the last output from both the directions
                        # out = self.fc(torch.concat([out[:,-1,:2],out[:,0,2:]],1)) # Takes last pred from both sides ????
                        ### Or take let attention optimize either together or separate where the forward and backward should be looking
                        # out1 = self.gru_attention1(out[:,:,:2])
                        # out2 = self.gru_attention2(out[:,:,2:])
                        # out = self.fc(torch.concat([out1, out2], 1))
                        return torch.nn.functional.normalize(torch.squeeze(out), p=2, dim=1)

                    # ... and you can also redefine the forward method,
                    # as you would for a typical pytorch model
                    def get_offset(self):
                        return cebra.data.Offset(self.leftset, self.rightset)
            elif architecture == "offset9UPDRS-model":
                @cebra.models.register("offset9UPDRS-model")  # --> add that line to register the model!
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
                        self.conv1 = nn.Conv1d(num_neurons - 1, 64, 2)
                        self.GELU = nn.GELU()
                        self.BN = nn.LazyBatchNorm1d()  # Can do some work to figure out actual sizes and not use Lazy
                        self.BN2 = nn.LazyBatchNorm1d()
                        self.BN3 = nn.LazyBatchNorm1d()
                        self.BN4 = nn.LazyBatchNorm1d()
                        self.BN5 = nn.LazyBatchNorm1d()
                        self.LN = nn.LayerNorm([64, 8])
                        self.LN2 = nn.LayerNorm([64, 7])
                        self.LN3 = nn.LayerNorm([64, 5])
                        self.LN4 = nn.LayerNorm([8, 3])
                        # self.LN5 = nn.LayerNorm([32, 8])
                        self.conv2 = nn.Conv1d(64, 64, 2)
                        self.conv3 = nn.Conv1d(64, 64, 3)
                        # self.conv4 = nn.Conv1d(64, 64, 3)
                        self.conv4alt = nn.Conv1d(64, 8, 3)
                        # self.skipconv2 = cebra_layers._Skip(self.conv2, self.GELU, self.LN3)
                        self.skipconv3 = cebra_layers._Skip(self.conv3, self.GELU, self.LN3)
                        # self.skipconv4 = cebra_layers._Skip(self.conv4,self.GELU,self.LN4)
                        # self.convout = nn.Conv1d(32, num_output, 3)
                        self.flat = nn.Flatten()
                        self.linint = nn.LazyLinear(32)
                        self.linout = nn.LazyLinear(num_output)
                        # nn.LazyLinear(32*3),
                        # nn.GELU(), # num_units*kernel size? (= channels * kernel)
                        # nn.LazyBatchNorm1d(),
                        # nn.Dropout(0.5),
                        # nn.LazyLinear(3),

                    def init_weights(m):
                        if isinstance(m, nn.Conv2d):
                            torch.nn.init.xavier_uniform_(m.weight)
                            with torch.no_grad():
                                m.bias.zero_()

                    def forward(self, x):
                        UPDRS = x[:, -1, 4]  # Take the UPDRS score of the sample corr to the label
                        x = x[:, 0:-1, :]
                        # x = self.drop02(x)
                        x = self.BN(self.GELU(self.conv1(x)))  # In: 512,36,9, Out: 512,32,8
                        # x = self.drop02(x)
                        x = self.BN2(self.GELU(self.conv2(x)))  # Out: 512,64,7
                        # Note that these are cropped skip connections
                        x = self.skipconv3(x)  # Out: 512,64,5
                        x = self.BN4(self.GELU(self.conv4alt(x)))  # Out: 512,64,3
                        x = self.flat(x)  # Out: 512,192
                        # x = self.BN5(self.GELU(self.linint(x))) # Out: 512,32
                        # Add the UPDRS score as a feature alongside x (now 32) # of other processed
                        x = self.GELU(self.linint(torch.concat((x, UPDRS.expand((1, x.size(dim=0))).T), 1)))  # UPDRS
                        x = self.linout(x)  # Out: 512,3
                        return torch.squeeze(x)

                    # ... and you can also redefine the forward method,
                    # as you would for a typical pytorch model
                    def get_offset(self):
                        return cebra.data.Offset(4, 5)
            elif architecture == "kernel9-model":
                @cebra.models.register("kernel9-model")  # --> add that line to register the model!
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
                        return cebra.data.Offset(4 + 9,
                                                 5 + 9)  # such that the full kernel can go on the full mov data >90% of times (90% at least 9 samples)

    def _get_continuous_color(self,colorscale, intermed):
        """
        Internal function:
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

    def _saveresultstoTensorBoard(self, mean_ba, batotal):
        '''Internal function to save the results stored in "p_" to numpy and those in "writer" to TensorBoard after the run'''
        # After the run
        self.p_['total_mean_ba'] = mean_ba
        self.p_['total_std'] = np.std(batotal)
        np.save(
            f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA performances/{self.curtime}_{self.val_approach}.npy",
            self.p_,
        )
        self._plot_results(save=True)
        # Calculate some final performance metric and save to TensorBoard
        metric_dict = {'mean_accuracy': mean_ba}
        # Calculate the mean of means (i.e. the performance mean ignoring imbalances in cohort sizes)
        ba_mean_ba = np.mean(self.bacohortlist)
        metric_dict['cohortbalanced_mean_accuracy'] = ba_mean_ba
        for coh in range(len(cohorts)):
            metric_dict[f'mean_{cohorts[coh]}'] = self.bacohortlist[coh]
        # Weird way of saving required to combine model parameters and metrics as hparams to allow visualizing their influences
        exp, ssi, sei = hparams(model_params, metric_dict)
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.writer.add_scalar(k, v)
        medloss = np.median(self.alllosses, axis=0)
        medtemp = np.median(self.alltemps, axis=0)
        for it in range(model_params['max_iterations']):
            self.writer.add_scalar('Median_loss', medloss[it], it)
            self.writer.add_scalar('Median_temp', medtemp[it], it)

    def _analyze_featureattentionModule(self):
        '''Internal function that can be used to analyze featureattentionModule, can be made public allowing passing a custom model (on the same data)'''
        scales = self.cebra_model.model.savedscales.detach().cpu().numpy()
        if scales.shape[2] > 1:
            meanscale = np.mean(scales, 0)
            g = sns.heatmap(meanscale, cmap="viridis")
            g.set_title("Learned Time - Feature scaling")
            g.set_yticks(np.arange(len(self.featurelist)) + 0.5)
            g.set_yticklabels(self.featurelist, rotation=0)
            for tick_label in g.axes.get_xticklabels():
                if tick_label.get_text() == '4':
                    tick_label.set_color("red")
            plt.show()

        else:
            scales = scales[:, :, 0]
            med = np.median(scales, 0)
            sorted_desc = np.argsort(med)[::-1]
            g = sns.boxplot(scales[:, sorted_desc])
            g.set_title("Scaling applied by the feature attention module [0,1]")
            g.set_xticklabels(self.featurelist[sorted_desc], rotation=45, ha='right')
            plt.show()

    def _plot_results(self, save=False,show=False):
        '''Internal function used to plot the results and optionally save or show the results; depends on the internal "p_" database,
        could be made external by allowing passing a custom p_?'''
        Col = ['Cohort', 'Validation', 'Performance']
        # Repeat this over
        df = pd.DataFrame(columns=['Cohort', 'Validation', 'Performance'])
        for cohort in self.cohorts:
            result = [[cohort, self.val_approach, self.p_[cohort][item]['performance']] for item in self.p_[cohort] if isinstance(self.p_[cohort][item],dict)]
            df_temp = pd.DataFrame(data=result, columns=Col)
            df = pd.concat([df, df_temp])

        # sns.boxplot(x="Cohort", y="Performance", data=df, saturation=0.5)
        sns.catplot(data=df, x="Validation", y="Performance", kind="box", color="0.9")
        sns.swarmplot(x="Validation", y="Performance", data=df, hue="Cohort",
                      size=4, edgecolor="black", dodge=True)
        plt.ylim(0.3, 1)
        if show:
            plt.show()
        if save:
            self.writer.add_figure('Performance_Figure', plt.gcf(), 0)

    def _create_Captum_networks(self):
        class LogisticRegressionNetwork(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(LogisticRegressionNetwork, self).__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                nn.init.uniform_(self.linear.weight, -0.01, 0.01)
                nn.init.zeros_(self.linear.bias)

            def forward(self, x):
                outputs = self.linear(x)
                return outputs

        class DirectClassifier(nn.Module):  # Concatenate the learned models in 1 model
            def __init__(self, CEBRA, Logres):
                super(DirectClassifier, self).__init__()
                self.modelA = CEBRA
                self.modelB = Logres

            def forward(self, x):
                x = self.modelA(x)
                x = self.modelB(x)
                return torch.sigmoid(x)
        return LogisticRegressionNetwork, DirectClassifier

    def get_patients_train_dict(self, sub_test, cohort_test, data_select: dict, val_approach = None):
        '''Function that takes current test subject, belonging cohort the dataset to use and for external use
        validation approach to generate a list of training data for that validation approach'''
        if not val_approach: # Can retrieve the train dict from the outside given the val_approach
            val_approach = self.val_approach
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

    def get_data_sub_ch(self, cohort, sub, ch, UPDRS=None):
        '''Function that gets the data from 1 channel of a subject from a specific cohort and optionally finds the updrs
        score and adds that as a feature'''
        X_train = []
        y_train = []

        for f in self.ch_all_feat[cohort][sub][ch].keys():
            X_train.append(self.ch_all_feat[cohort][sub][ch][f]["data"])
            y_train.append(self.ch_all_feat[cohort][sub][ch][f]["label"])

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

    def get_data_channels(self,sub_test: str, cohort_test: str, df_rmap: pd.DataFrame, naiveupdrs=False):
        '''Function that selects the channel from the r_map and then uses get_data_sub_ch to gather the data'''
        if sub_test != '000':
            sub_num = sub_test.lstrip('0')
        else:
            sub_num = '0'
        ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_num")[
            "ch"
        ].iloc[0]
        if not naiveupdrs:
            X_test, y_test = self.get_data_sub_ch(cohort_test, sub_test, ch_test
            )
        else:
            UPDRS = self.df_updrs.query("cohort == @cohort_test and sub == @sub_test")['UPDRS_total'].iloc[0]
            if np.isnan(UPDRS): # TODO: Properly fix that not all subjects have UPDRS
                UPDRS = 41.21052632 # QUICK PATCHUP TO RUN A TEST
            X_test, y_test = self.get_data_sub_ch(cohort_test, sub_test, ch_test, UPDRS
            )
        return X_test, y_test

    def collect_training_data(self, cohorts_train):
        '''Function that takes the training dictionary and concatenates all the data'''
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
                X_train, y_train = self.get_data_channels(
                    sub_train, cohort_train, df_rmap=self.df_best_rmap
                )
                sub_aux = np.tile(sub_counter, len(y_train))

                y_train_discr_comb.append(np.squeeze(y_train))  # Save the true labels
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
        return X_train, y_train, y_train_discr, sub_aux, coh_aux

    def plotly_embeddings(self,X_train_emb,y_train_discr,X_test_emb,y_test,test_coh,aux=None,type='none', nearest=True,model=None,val=None):
        '''Function that uses Plotly to visualize the embeddings, overlaying the training and test embedding and labeling them by their class,
        includes methods to color the cohorts, subjects or none of those
        Also contains a method that can show the distance to movement onset / or to closest movement.
        Parameters:
            aux: The current auxillary variable used
            type: Whether to color cohorts, subjects, none or the distance to movements ("coh","sub","none","grad")
            nearest: True or False, in combination with type="grad" sets if data is colored based on distance to movement onset (False)
                or nearest movement in general (True)
            model: For logistic regression decoder can be used to visualize the decoding boundaru, currently used only for aux="coh"
            val: The current validation approach: used to create the correct legend for type="coh" coloring

            TODO:
            Extend the plotting functionality of the other plot types besides "aux"
            Check if other types besides "aux" still function correctly
            '''
        #Plot together with the test embedding (For coh auxillary)
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
            colorlist = [self._get_continuous_color(colorscale, intermed=0.25),self._get_continuous_color(colorscale, intermed=0.75),self._get_continuous_color(colorscale, intermed=0.0),self._get_continuous_color(colorscale, intermed=1)]

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
            # Separate plot per label such that the legend is correct
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

            # Legacy plotter, simpler but less detailed
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
        elif type == 'grad':
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

    def build_and_fit_cebra(self,X_train,y_train,y_train_discr,coh_aux):
        '''Function that builds and fits the CEBRA model using the PyTorch API, following specifications
        given in "model_params"'''
        # Put in TensorDataset with 2d discrete in order cohort - movement
        # Put in TensorDataset with 2d discrete in order cohort - movement
        CohortAuxData = cebra.data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),
                                                 discrete=torch.from_numpy(
                                                     np.array([coh_aux, y_train], dtype=int).T).type(
                                                     torch.LongTensor)).to('cuda')

        self._register_model()
        self.neural_model = cebra.models.init(
            name=model_params['model_architecture'],
            num_neurons=CohortAuxData.input_dimension,
            num_units=32,
            num_output=model_params['output_dimension'],
            latent=model_params['latent'],
            numlayers=model_params['numlayers'],
            left_set=model_params['left_set'],
        ).to('cuda')

        CohortAuxData.configure_for(self.neural_model)
        Crit = cebra.models.criterions.LearnableCosineInfoNCE(temperature=model_params['temperature'],
                                                              min_temperature=model_params[
                                                                  'min_temperature']).to('cuda')

        Opt = torch.optim.Adam(list(self.neural_model.parameters()) + list(Crit.parameters()),
                               lr=model_params['learning_rate'], weight_decay=0)
        self.cebra_model = cebra.solver.init(name="single-session", model=self.neural_model, criterion=Crit, optimizer=Opt,
                                        tqdm_on=True).to(
            'cuda')
        Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                          batch_size=model_params['batch_size'], prior=model_params['prior'],
                                          cond=model_params['conditional']).to('cuda')
        # Compute some derivatives
        self.offset = self.neural_model.get_offset()
        # Maybe find a better place to do this
        # Defines how to slice the data without using padding
        self.rightbound = -self.offset.right + 1
        if self.rightbound == 0:
            self.y_train_offset = y_train_discr[self.offset.left:]
            self.coh_aux_offset = coh_aux[self.offset.left:]
        else:
            self.y_train_offset = y_train_discr[self.offset.left:self.rightbound]
            self.coh_aux_offset = coh_aux[self.offset.left:self.rightbound]

        Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                          batch_size=model_params['batch_size'], prior=model_params['prior'],
                                          cond=model_params['conditional']).to('cuda')
        self.cebra_model.fit(loader=Loader)
        if type(self.alllosses) == int:
            self.alllosses = np.array(np.expand_dims(self.cebra_model.state_dict()["loss"], axis=0))
            self.alltemps = np.array(np.expand_dims(self.cebra_model.state_dict()["log"]["temperature"], axis=0))
        else:
            self.alllosses = np.concatenate((self.alllosses, np.expand_dims(self.cebra_model.state_dict()["loss"], axis=0)), axis=0)
            self.alltemps = np.concatenate((self.alltemps, np.expand_dims(self.cebra_model.state_dict()["log"]["temperature"], axis=0)), axis=0)

    def build_and_fit_cebra_earlystop(self,X_train,y_train,y_train_discr,coh_aux):
        '''IN DEVELOPMENT: Function that builds and fits the CEBRA model using the PyTorch API, following specifications
        given in "model_params" but includes test data as validation split to test if optimizing for InfoNCE using a sub
        split would be beneficial

        TODO:
         Make the validation set all of the subjects for leave-cohort-out (else early stop on 1 only) OR change to train separate per sub (more accurate)
         '''
        # Put in TensorDataset with 2d discrete in order cohort - movement
        CohortAuxData = cebra.data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),
                                                 discrete=torch.from_numpy(
                                                     np.array([coh_aux, y_train], dtype=int).T).type(
                                                     torch.LongTensor)).to('cuda')
        self._register_model()

        self.neural_model = cebra.models.init(
            name=model_params['model_architecture'],
            num_neurons=CohortAuxData.input_dimension,
            num_units=32,
            num_output=model_params['output_dimension']
        ).to('cuda')

        CohortAuxData.configure_for(self.neural_model)
        Crit = cebra.models.criterions.LearnableCosineInfoNCE(temperature=model_params['temperature'],
                                                              min_temperature=model_params[
                                                                  'min_temperature']).to('cuda')

        Opt = torch.optim.Adam(list(self.neural_model.parameters()) + list(Crit.parameters()),
                               lr=model_params['learning_rate'], weight_decay=0)
        self.cebra_model = cebra.solver.init(name="single-session", model=self.neural_model, criterion=Crit,
                                             optimizer=Opt,
                                             tqdm_on=True).to(
            'cuda')
        Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                          batch_size=model_params['batch_size'], prior=model_params['prior'],
                                          cond=model_params['conditional']).to('cuda')
        # Compute some derivatives
        self.offset = self.neural_model.get_offset()
        # Maybe find a better place to do this
        # Defines how to slice the data without using padding
        rightbound = -self.offset.right + 1
        if rightbound == 0:
            self.y_train_offset = y_train_discr[self.offset.left:]
            self.y_test_offset = self.y_test[self.offset.left:]
            self.coh_aux_offset = coh_aux[self.offset.left:]
        else:
            self.y_train_offset = y_train_discr[self.offset.left:rightbound]
            self.y_test_offset = self.y_test[self.offset.left:rightbound]
            self.coh_aux_offset = coh_aux[self.offset.left:rightbound]

        Loader = CohortDiscreteDataLoader(dataset=CohortAuxData, num_steps=model_params['max_iterations'],
                                          batch_size=model_params['batch_size'], prior=model_params['prior'],
                                          cond=model_params['conditional']).to('cuda')
        ValidData = cebra.data.TensorDataset(torch.from_numpy(self.X_test).type(torch.FloatTensor),
                                             discrete=torch.from_numpy(
                                                 np.array([np.repeat(0, len(self.y_test)), self.y_test],
                                                          dtype=int).T).type(
                                                 torch.LongTensor)).to('cuda')
        ValidData.configure_for(self.neural_model)
        Valid_loader = CohortDiscreteDataLoader(dataset=ValidData, num_steps=1,
                                                batch_size=len(self.X_test) - len(self.X_test) % 8).to('cuda')
        self.cebra_model.fit(loader=Loader, valid_loader=Valid_loader, save_frequency=5, valid_frequency=5,
                             logdir="C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRAsaves")
        # Save the total losses first before resetting to the early stopping
        if type(self.alllosses) == int:
            self.alllosses = np.array(np.expand_dims(self.cebra_model.state_dict()["loss"], axis=0))
            self.alltemps = np.array(np.expand_dims(self.cebra_model.state_dict()["log"]["temperature"], axis=0))
        else:
            self.alllosses = np.concatenate(
                (self.alllosses, np.expand_dims(self.cebra_model.state_dict()["loss"], axis=0)), axis=0)
            self.alltemps = np.concatenate(
                (self.alltemps, np.expand_dims(self.cebra_model.state_dict()["log"]["temperature"], axis=0)), axis=0)
        # Load model with lowest validation loss (INFONCE on validation data)
        self.cebra_model.load(logdir="C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRAsaves", filename='checkpoint_best.pth')

    def get_embedding(self,X_data):
        '''Function to generate embeddings from a dataset (train or test)'''
        TrainBatches = np.lib.stride_tricks.sliding_window_view(X_data, self.offset.__len__(), axis=0)
        X_emb = self.cebra_model.transform(
            torch.from_numpy(TrainBatches[:]).type(torch.FloatTensor).to('cuda')).to('cuda')
        X_emb = X_emb.cpu().detach().numpy()
        return X_emb

    def appendembeddings(self,embeddings,embeddinglabels,ID,X_train_emb,y_train_offset,cohort_test): # Merge these two and add true false for train test / or do at once?
        '''Function to append embeddings to a list for use in computing the embedding consistency'''
        embeddings.append(X_train_emb)
        ID.append('+'.join([coh for coh in cohorts if coh != cohort_test]))
        embeddinglabels.append(y_train_offset)
        return embeddings, embeddinglabels, ID

    def trainClassifier(self,trainemb,y_train,decmodel = None):
        '''Function that chooses a classifier that to decode movement from the embedding and train it;
        will use model_params['decoder'] to select if decmodel is left empty (could be used for external use)'''
        if not decmodel: # Allow use of this function without
            decmodel = self.model_params['decoder']
        if self.Captum:
            if decmodel == 'Logistic':
                if not 'LogisticRegressionNetwork' in dir():
                    self.LogisticRegressionNetwork, self.DirectClassifier = self._create_Captum_networks()
                decoder = self.LogisticRegressionNetwork(3,1)
                decoder.train(True)
                Opt_log = torch.optim.LBFGS(list(decoder.parameters()))
                pos_w = np.sum((len(y_train)-np.sum(y_train))/(np.sum(y_train)))
                weights = np.zeros(np.shape(y_train))
                weights += 1
                weights[np.array(y_train,dtype=bool)] = pos_w
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights)) # TODO: Calculate pos_weight
                def closure(): # Define closure
                    Opt_log.zero_grad()
                    outputs = decoder(torch.Tensor(trainemb))
                    loss = loss_fn(torch.squeeze(outputs),
                                   torch.Tensor(y_train))
                    loss.backward()
                    return loss
                for i in range(100): # 100 optimization steps like sklearn
                    # Adjust learning weights
                    Opt_log.step(closure)
            else:
                raise ValueError(f'Only Logistic Regression is currently supported with XAI/Captum, you chose {decmodel}')
        else:
            if decmodel == 'KNN':
                decoder = neighbors.KNeighborsClassifier(
                n_neighbors=model_params['n_neighbors'], metric=model_params['metric'],
                n_jobs=model_params['n_jobs'])
                decoder.fit(trainemb, np.array(y_train, dtype=int))
            elif decmodel == 'Logistic':
                decoder = linear_model.LogisticRegression(class_weight='balanced', penalty=None)
                decoder.fit(trainemb, np.array(y_train, dtype=int))
            elif decmodel == 'SVM':
                decoder = SVC(kernel=cosine_similarity)
                decoder.fit(trainemb, np.array(y_train, dtype=int))
            elif decmodel == 'FAISS':
                raise Exception('Not implemented')
            elif decmodel == 'MLP':
                raise Exception('Not implemented')
            elif decmodel == 'XGB':
                decoder = xgboost.sklearn.XGBClassifier()
                # decoder.set_params(**{'lambda':2})
                classes_weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
                decoder.set_params(eval_metric="auc")
                decoder.fit(
                    trainemb,
                    np.array(y_train, dtype=int),
                    sample_weight=classes_weights,
                )
            elif decmodel == 'KNN_BPP':
                decoder = kNN_BPP(n_neighbors=model_params['n_neighbors'])
                decoder.fit(trainemb, np.array(y_train, dtype=int))
            else:
                print(f"selected decoder model {decmodel} is not available")
                raise
        return decoder

    def computeconsistency(self,embeddings):
        '''IN DEVELOPMENT: Function that computes the consistency between embeddings in a list

        TODO:
        Find a way to compute the consistency'''
        # Maybe use iterative closest point analysis, or see how the dataset method can be used
        # The dataset method bins based on label and then matches the mean of each label, This is not nice with
        # only 2 means (undefined)
        scores_sub, pairs_sub, datasets_sub = cebra.sklearn.metrics.consistency_score(embeddings=embeddings,
                                                                                      # dataset_ids=testID,
                                                                                      between="runs")
        plt.figure()
        cebra.plot_consistency(scores_sub, pairs_sub, datasets_sub, vmin=0, vmax=100,
                               title="Between-subject consistencies")
        plt.show()

    def ExplainableAI(self,X_test_batch,X_train_batch):
        '''Runs some explainable AI methods from the Captum package on the CEBRA model to gain insight into feature importances

        TODO: Add short description of the different methods
        '''
        # Do the XAI stuff and compute feature importances
        torch.backends.cudnn.enabled = False
        compmodel = self.DirectClassifier(self.cebra_model.model, self.decoder.cuda())
        del self.cebra_model
        del self.decoder

        ig = IntegratedGradients(compmodel)
        ig_nt = NoiseTunnel(ig)
        # dl = DeepLift(compmodel)
        gs = GradientShap(compmodel)
        fa = FeatureAblation(compmodel)

        ig_attr_test = ig.attribute(X_test_batch, n_steps=50)
        ig_attr_test = ig_attr_test.detach().cpu().numpy().sum(0)
        ig_attr_test_norm_sum = ig_attr_test / np.linalg.norm(ig_attr_test, ord=1)
        ig_nt_attr_test = ig_nt.attribute(X_test_batch, n_steps=40)
        # dl_attr_test = dl.attribute(X_test_batch)
        gs_attr_test = gs.attribute(X_test_batch, X_train_batch)
        fa_attr_test = fa.attribute(X_test_batch)
        # prepare attributions for visualization

        x_axis_data = np.arange(X_test_batch.shape[1])
        x_axis_data_labels = list(map(lambda idx: self.featurelist[idx], x_axis_data))

        ig_nt_attr_test_sum = ig_nt_attr_test.detach().cpu().numpy().sum(0)
        ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

        # dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
        # dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

        gs_attr_test_sum = gs_attr_test.detach().cpu().numpy().sum(0)
        gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

        fa_attr_test_sum = fa_attr_test.detach().cpu().numpy().sum(0)
        fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

        width = 0.14
        legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'GradientSHAP', 'Feature Ablation']

        plt.figure(figsize=(20, 10))

        ax = plt.subplot()
        ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
        ax.set_ylabel('Attributions')

        FONT_SIZE = 16
        plt.rc('font', size=FONT_SIZE)  # fontsize of the text sizes
        plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

        ax.bar(x_axis_data, ig_attr_test_norm_sum[:, -1], width, align='center', alpha=0.8, color='#eb5e7c')
        ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum[:, -1], width, align='center', alpha=0.7, color='#A90000')
        ax.bar(x_axis_data + 2 * width, gs_attr_test_norm_sum[:, -1], width, align='center', alpha=0.8, color='#4260f5')
        ax.bar(x_axis_data + 3 * width, fa_attr_test_norm_sum[:, -1], width, align='center', alpha=1.0, color='#49ba81')
        ax.autoscale_view()
        plt.tight_layout()

        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_data_labels, rotation=45, ha='right')

        plt.legend(legends, loc=3)
        plt.show()
        torch.backends.cudnn.enabled = True
    def run_CV(self,val_approach='',show_embedding=False, embeddingconsistency=False,showfeatureweights=False,Testphase=False,Captum=True):
        ''''The main cross validation loop. Will use the self.val_approach as determinant on what approach to run,
        or can be manually set to run a certain validation approach and give the results.

        TODO:
        Saving the result (with model_params["debug"] == False) might require defining self.writer if not run through loop_approaches
        Properly handle the auxillary parameters checking if they have been set by the loop, else using the default setting
        '''
        if not val_approach: # allow running CV externally with given val_approach
            val_approach = self.val_approach
            if not val_approach:
                raise ValueError('No validation approach defined')
        if not self.writer and not self.model_params['debug']:
            self.curtime = datetime.now().strftime("%Y_%m_%d-%H_%M")
            self.writer = SummaryWriter(
                log_dir=f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA_logs/{self.val_approach}/{self.curtime}")
        # Very local parameters
        trainembeddings = []
        trainID = []
        trainembeddinglabels = []
        testembeddings = []
        testID = []
        testembeddinglabels = []
        cohort_prev_it = ""
        # Shared parameters that need to refresh each loop
        self.p_ = {}
        self.batotal = []
        self.bacohortlist = []
        self.cohort_prev_it = ""
        self.alllosses = 0
        self.alltemps = 0
        # Loop over test cohorts
        for cohort_test in cohorts:
            # Prepare some datastorage and select what which test subjects to use from the cohorts
            if cohort_test not in self.p_:
                self.p_[cohort_test] = {}
            bacohort = []
            # Determine test subjects
            if self.Testphase:
                subtests = self.ch_all_test[cohort_test].keys()
            else:
                subtests = self.ch_all_train[cohort_test].keys()
            # Loop over these test subjects
            for sub_test in subtests:

                print('Val approach, cohort, subject:', val_approach, cohort_test, sub_test)
                # Create entry in results dict
                if sub_test not in self.p_[cohort_test]:
                    self.p_[cohort_test][sub_test] = {}
                # Get test data
                self.X_test, self.y_test = self.get_data_channels(
                    sub_test, cohort_test, df_rmap=self.df_best_rmap
                )
                # Select the desired features
                self.X_test = self.X_test[:, self.idxlist].copy()
                # if statement to not retrain model within a cohort for leave 1 out cohort procedure
                if (val_approach == "leave_1_cohort_out" and cohort_test != cohort_prev_it) or val_approach != "leave_1_cohort_out":
                    # Select what training data can be used (all or only validation split)
                    if not self.Testphase:
                        cohorts_train = self.get_patients_train_dict(
                            sub_test, cohort_test, data_select=self.ch_all_train
                        )
                    else:
                        cohorts_train = self.get_patients_train_dict(
                            sub_test, cohort_test, data_select=self.ch_all
                            # Use all data in training (except test data, which gets rejected in the function)
                        )
                    # Collect training data
                    self.X_train, self.y_train, self.y_train_discr, self.sub_aux, self.coh_aux = self.collect_training_data(cohorts_train)
                    # Select features
                    self.X_train = self.X_train[:,self.idxlist]
                    # Build and fit CEBRA model depending on if earlystopping is used
                    if self.model_params['early_stopping']:
                        self.build_and_fit_cebra_earlystop(self.X_train,self.y_train,self.y_train_discr,self.coh_aux)
                    else:
                        self.build_and_fit_cebra(self.X_train,self.y_train,self.y_train_discr,self.coh_aux)
                    # Generate the embedding using the model
                    self.X_train_emb = self.get_embedding(self.X_train)

                    # Plot feature weights
                    if self.showfeatureweights:
                        self._analyze_featureattentionModule()
                    # Append the embeddings to allow comparison later
                    if self.embeddingconsistency:
                        trainembeddings = self.appendembeddings(trainembeddings,trainembeddinglabels,trainID,self.X_train_emb,self.y_train_offset,cohort_test)
                    # Train the decoder on the embedding
                    self.decoder = self.trainClassifier(self.X_train_emb,self.y_train_offset) # pass the data or as self

                # Update what the cohort was in the previous iteration
                cohort_prev_it = cohort_test
                # TEST PERMUTATION OF FEATURES
                # rng = np.random.default_rng()
                # self.X_test_emb = cebra_model.transform(rng.permutation(self.X_test,axis=1),session_id=0)

                # Predict embedding for the test data
                self.X_test_emb = self.get_embedding(self.X_test)
                # Cut of the boundaries as we do not use padding
                if self.rightbound == 0:
                    self.y_test_offset = self.y_test[self.offset.left:]
                else:
                    self.y_test_offset = self.y_test[self.offset.left:self.rightbound]
                # Append test to embedding list for consistency calculations
                if self.embeddingconsistency:
                    testembeddings = self.appendembeddings(testembeddings,testembeddinglabels,testID,self.X_test_emb,self.y_test_offset,cohort_test)
                # Plot the embeddings if desired
                if self.show_embedding:
                        self.plotly_embeddings(self.X_train_emb,self.X_test_emb,self.y_train_offset,
                                      self.y_test_offset,cohort_test,aux=self.coh_aux_offset,type='coh', grad=False, nearest=False,model=None,val=val_approach)
                # Predict the class based on the test embedding and the trained decoder
                if not self.Captum:
                    self.y_test_pr = self.decoder.predict(self.X_test_emb)
                else:
                    self.decoder.eval()
                    self.y_test_pr = self.decoder(torch.Tensor(self.X_test_emb))
                    self.y_test_pr = torch.sigmoid(self.y_test_pr) > 0.5  # Have to add sigmoid now and binarize
                    self.y_test_pr = self.y_test_pr.detach().cpu().numpy()

                # Compute balanced accuracy and store the results in dict
                ba = metrics.balanced_accuracy_score(self.y_test_offset, self.y_test_pr)
                print(ba)
                # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(self.X_test_emb))

                self.p_[cohort_test][sub_test] = {}
                self.p_[cohort_test][sub_test]["performance"] = ba
                self.p_[cohort_test][sub_test]["y_test"] = self.y_test_offset
                self.p_[cohort_test][sub_test]["y_test_pr"] = self.y_test_pr
                self.p_[cohort_test][sub_test]["loss"] = self.cebra_model.state_dict()["loss"]
                self.p_[cohort_test][sub_test]["temp"] = self.cebra_model.state_dict()["log"]["temperature"]

                # Save performance metrics over the loops
                bacohort.append(ba)
                self.batotal.append(ba)

                if self.Captum:
                    TestBatches = np.lib.stride_tricks.sliding_window_view(self.X_test, self.offset.__len__(), axis=0)
                    X_test_batch = torch.from_numpy(TestBatches).type(torch.FloatTensor).to('cuda')
                    TrainBatches = np.lib.stride_tricks.sliding_window_view(self.X_train, self.offset.__len__(), axis=0)
                    X_train_batch = torch.from_numpy(TrainBatches).type(torch.FloatTensor).to('cuda')
                    self.ExplainableAI(X_test_batch,
                                       X_train_batch)

            # After all subjects
            self.bacohortlist.append(np.mean(bacohort))
            mean_ba = np.mean(self.batotal)
            print(f'running mean balanced accuracy: {mean_ba}')
            self.p_[cohort_test]['mean_ba'] = np.mean(bacohort)
            self.p_[cohort_test]['std_ba'] = np.std(bacohort)

            # Compute for each of the training runs the consistency between the test subject embeddings
            if self.embeddingconsistency: # For each of the training
                self.computeconsistency(testembeddings)
        # After all cohorts
        # Compute the consistency between training embeddings
        if self.embeddingconsistency:
            self.computeconsistency(trainembeddings)
        # Save the performance dict and write to TensorBoard if it was not a debug run
        if not model_params['debug']:
            self._saveresultstoTensorBoard(mean_ba,self.batotal)

    def loop_approaches(self,show_embedding=False,embeddingconsistency=False, showfeatureweights=False, Testphase=False,Captum=False):
        '''Function that loops over all validation approaches given in self.val_approaches and runs the cross validation for each approach.
        Parameters:
            show_embedding: whether to show the visualize the resulting embedding using Plotly; default=False
            embeddingconsistency: IN DEVELOPMENT whether to compute the consistency between embeddings (training and test embeddings);
                                    default=False
            showfeatureweights: whether to visualize the weights of an attention module in the neural model, make sure
                                an attention_model is present.; default=False
            Testphase: whether the model is in test mode, calculating the accuracy only based on left-out test subjects
                        default = False (train/validation mode)
            Captum: Set to True to use explainable AI methods to gain insight into feature importances'''
        self.show_embedding = show_embedding
        self.embeddingconsistency = embeddingconsistency
        self.showfeatureweights = showfeatureweights
        self.Testphase = Testphase
        self.Captum = Captum
        self.curtime = datetime.now().strftime("%Y_%m_%d-%H_%M")
        for val_approach in self.val_approaches:
            self.val_approach = val_approach
            if not self.model_params['debug']:
                self.writer = SummaryWriter(log_dir=f"C:/Users/ICN_GPU/Documents/Glenn_Data/CEBRA_logs/{self.val_approach}/{self.curtime}")
            t0 = time.time()
            self.run_CV()
            print(f'runtime: {time.time() - t0}')

# MAIN

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
            'conditional':'mov', # Set to mov or cohmov to either equalize reference-positive in movement or cohort and movement
            'early_stopping':False,
            'features': 'Hjorth,fft,Sharpwave,fooof,bursts', # Choose what features to include 'Hjorth,fft,Sharpwave,fooof,bursts' as 1 string separated by commas without spaces
            'latent': 2,
            'numlayers': 1,
            'left_set': 9,
            'additional_comment':'test_AttentionWithContext',
            'debug': True} # Debug = True; stops saving the results unnecessarily

val_approaches = ["leave_1_sub_out_within_coh"]
cohorts = ["Pittsburgh","Berlin","Beijing","Washington"]
search_space = list()
search_space.append(Integer(200, 1000, name='max_iterations'))
search_space.append(Categorical([256, 512, 1028], name='batch_size'))
search_space.append(Integer(3, 12, name='output_dimension'))
search_space.append(Real(0.1,1,name='min_temperature'))
search_space.append(Categorical(['fft','Hjorth,fft,Sharpwave,fooof,bursts'], name='features'))
search_space.append(Integer(1,12, name='latent'))
search_space.append(Integer(1,3, name='numlayers'))
search_space.append(Integer(3,15, name='left_set'))

global it
it = 1
allbas = []
@use_named_args(search_space)
def evaluate_model(**params):
    global it
    print(it)
    it+=1
    model_params.update(params)
    run1 = run_cross_val_cebra(model_params, val_approaches, cohorts)  # Put in the data
    run1.loop_approaches(show_embedding=False, embeddingconsistency=False,showfeatureweights=False,Testphase=False,Captum=False)
    # Save for every model all the performances such that the performances can be plot for the best model
    allbas.append(run1.batotal)
    # Optimized on the average balanced accuracy
    ba_avg = np.mean(run1.batotal)
    return 1.0-ba_avg

result = gp_minimize(evaluate_model, search_space)
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: %s' % (result.x))