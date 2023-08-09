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

model = cebra.models.model.Offset10Model(10, 10, 4, normalize=True)
#model = cebra.models.model.SupervisedNN10(10, 5, 4)
solver = cebra.solver.multi_session.MultiSessionSolver(model,
                                              cebra.models.criterions.LearnableCosineInfoNCE(temperature=1.0, min_temperature=None),
                                              torch.optim.SGD(model.parameters(),lr=0.1))

solver.f
