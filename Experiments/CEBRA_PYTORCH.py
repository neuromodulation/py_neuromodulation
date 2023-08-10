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
import torch
from torch import nn


session1 = torch.randn((1000, 30))
session2 = torch.randn((1000, 50))
index1 = torch.randn((1000, 4))
index1int = torch.randint(0,2,(1000, ))
index2 = torch.randn((1000, 4)) # same index dim as index1
Data = cebra.data.DatasetCollection(
              cebra.data.TensorDataset(session1, continuous=index1),
              cebra.data.TensorDataset(session2, continuous=index2))
Datasingle = cebra.data.TensorDataset(session1, discrete=index1int)

#model = cebra.models.model.Offset10Model(Data.input_dimension, 10, 4, normalize=True)
#model = cebra.models.model.SupervisedNN10(10, 5, 4)

model = nn.ModuleList([
    cebra.models.init(
        name="offset10-model",
        num_neurons=dataset.input_dimension,
        num_units=32,
        num_output=8,
    ) for dataset in Data.iter_sessions()]).to("cuda")

modelsingle = cebra.models.init(
        name="offset5-model",
        num_neurons=Datasingle.input_dimension,
        num_units=32,
        num_output=8
    ).to("cuda")

Crit = cebra.models.criterions.LearnableCosineInfoNCE(temperature=1.0, min_temperature=None)
Opt = torch.optim.SGD(model.parameters(),lr=0.1)

solver = cebra.solver.init(name="single-session",model=modelsingle,criterion=Crit,optimizer=Opt)

#solver = cebra.solver.single_session.SingleSessionSolver(modelsingle,
#                                              Crit,
#                                              Opt)

#cebra.datasets.init('demo-continuous')
Loader = cebra.data.DiscreteDataLoader(dataset=Datasingle,num_steps=100,batch_size=25)

solver.fit(Loader)