import cebra
import torch
from torch import nn

session1 = torch.randn((100, 30))
session2 = torch.randn((100, 50))
index1 = torch.randn((100, 4))
index1int = torch.randint(0,2,(100, 4),dtype=torch.float32)
index1inttrue = torch.randint(0,2,(100, ))
index2 = torch.randn((100, 4)) # same index dim as index1
Data = cebra.data.DatasetCollection(
              cebra.data.TensorDataset(session1, continuous=index1),
              cebra.data.TensorDataset(session2, continuous=index2))
Datasingle = cebra.data.TensorDataset(session1, discrete=index1int, continuous=index2)

sampled = cebra.distributions.continuous.TimedeltaDistribution(index1int, time_delta=1, device='cpu', seed=None)
sampledDiscr = cebra.distributions.discrete.DiscreteEmpirical(index1inttrue, device='cpu', seed=None)
num_samples = 40
reference_idx = sampledDiscr.sample_prior(num_samples * 2)
negative_idx = reference_idx[num_samples:]
reference_idx = reference_idx[:num_samples]
reference = index1inttrue[reference_idx]
positive_idx = sampledDiscr.sample_conditional(reference)