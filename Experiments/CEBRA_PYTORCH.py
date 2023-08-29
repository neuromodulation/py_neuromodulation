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
import abc
import collections
from typing import List
from typing import Literal, Optional, Union
import numpy.typing as npt
import scipy.interpolate

import cebra.distributions.base as abc_

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex


session1 = torch.randn((1000, 30))
session2 = torch.randn((1000, 50))
index1 = torch.randn((1000, 4))
index1int = torch.randint(0,2,(1000, 2))
index2 = torch.randn((1000, 4)) # same index dim as index1
Data = cebra.data.DatasetCollection(
              cebra.data.TensorDataset(session1, continuous=index1),
              cebra.data.TensorDataset(session2, continuous=index2)).to('cuda') # Not sure if necessary to sent to device
Datasingle = cebra.data.TensorDataset(session1, discrete=index1int).to('cuda')

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
    ).to('cuda')

Crit = cebra.models.criterions.LearnableCosineInfoNCE(temperature=1.0, min_temperature=None).to('cuda')
Opt = torch.optim.Adam(list(modelsingle.parameters())+list(Crit.parameters()),lr=0.1)

solver = cebra.solver.init(name="single-session",model=modelsingle,criterion=Crit,optimizer=Opt).to('cuda')

#solver = cebra.solver.single_session.SingleSessionSolver(modelsingle,
#                                              Crit,
#                                              Opt)

Datasingle.configure_for(modelsingle)
#cebra.datasets.init('demo-continuous')

class CustomDiscrete(abc_.ConditionalDistribution, abc_.HasGenerator):
    """Resample multi-dimensional discrete data.

    The distribution is fully specified by an array of discrete samples.
    Samples can be drawn either from the dataset directly (i.e., output
    samples will have the same distribution of class labels as the samples
    used to specify the distribution), or from a resampled data distribution
    where the occurrence of each class label is balanced.

    Args:
        samples: Discrete index used for sampling
        samples: Discrete index used for sampling
    """

    def _to_numpy_int(self, samples: Union[torch.Tensor,
                                           npt.NDArray]) -> npt.NDArray:
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        if samples.dtype not in (np.int32, np.int64):
            samples = samples.astype(int)
        return samples

    def __init__(
        self,
        samples: torch.Tensor,
        device: Literal["cpu", "cuda"] = "cpu",
        seed: Optional[int] = None,
    ):
        abc_.HasGenerator.__init__(self, device=device, seed=None)
        self._set_data(samples)
        # self.sorted_idx = torch.from_numpy(np.argsort(self.samples))
        self.sorted_idx = torch.from_numpy(np.lexsort((self.samples[:,1],self.samples[:,0]))) # Multi dim version --> Sort first by 0th (cohort), then by movement 1st)
        self._init_transform()

    def _set_data(self, samples: torch.Tensor):
        samples = self._to_numpy_int(samples)
        #if samples.ndim > 1:
        #   raise ValueError(
        #        f"Data dimensionality is {samples.shape}, but can only accept a single dimension."
        #    )
        self.samples = samples

    @property
    def num_samples(self) -> int:
        """Number of samples in the index."""
        return len(self.samples) # TODO: Change to np.shape to get the correct dimension

    def _init_transform(self):
        # TODO: Create a CDF over the combined cohort-movement labels (sample uniformly across mov/no-move and cohorts)
        # TODO: Think of a different sampling method to simultaneously sample across cohorts and subjects
        self.counts = np.bincount(2*self.samples[:,0]+self.samples[:,1]) # Expects the cohorts in additive order (0,1,2...)
        self.cdf = np.zeros((len(self.counts) + 1,))
        self.cdf[1:] = np.cumsum(self.counts)
        # NOTE(stes): This is the only use of a scipy function in the entire code
        # base for now. Replacing scipy.interpolate.interp1d with an equivalent
        # function from torch would make it possible to drop scipy as a dependency
        # of the package.
        self.transform = scipy.interpolate.interp1d(
            np.linspace(0, self.num_samples, len(self.cdf)), self.cdf)

    def sample_uniform(self, num_samples: int) -> torch.Tensor:
        """Draw samples from the uniform distribution over values.

        This will change the likelihood of values depending on the values
        in the given (discrete) index. When reindexing the dataset with
        the returned indices, all values in the index will appear with
        equal probability.

        Args:
            num_samples: Number of uniform samples to be drawn.

        Returns:
            A batch of indices from the distribution. Reindexing the
            index samples of this instance with the returned in indices
            will yield a uniform distribution across the discrete values.
        """
        samples = np.random.uniform(0, self.num_samples, (num_samples,))
        samples = self.transform(samples).astype(int)
        return self.sorted_idx[samples]

    def sample_empirical(self, num_samples: int) -> torch.Tensor:
        """Draw samples from the empirical distribution.

        Args:
            num_samples: Number of samples to be drawn.

        Returns:
            A batch of indices from the empirical distribution,
            which is the uniform distribution over ``[0, N-1]``.
        """
        samples = np.random.randint(0, self.num_samples, (num_samples,))
        return self.sorted_idx[samples]

    def sample_conditional(self, reference_index: torch.Tensor) -> torch.Tensor:
        """Draw samples conditional on template samples.

        Args:
            samples: batch of indices, typically drawn from a
                prior distribution. Conditional samples will match
                the values of these indices

        Returns:
            batch of indices, whose values match the values
            corresponding to the given indices.
        """
        reference_index = self._to_numpy_int(2*reference_index[:,0]+reference_index[:,1])
        idx = np.random.uniform(0, 1, len(reference_index))
        idx *= self.cdf[reference_index + 1] - self.cdf[reference_index]
        idx += self.cdf[reference_index]
        idx = idx.astype(int)

        return self.sorted_idx[idx]

class CustomDiscreteUniform(CustomDiscrete, abc_.PriorDistribution):
    """Re-sample the given indices and produce samples from a uniform distribution."""

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        return self.sample_uniform(num_samples)


class CustomDiscreteEmpirical(CustomDiscrete, abc_.PriorDistribution):
    """Draw samples from the empirical distribution defined by the passed index."""

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        return self.sample_empirical(num_samples)
@dataclasses.dataclass
class CustomCohortDiscreteDataLoader(cebra_data.Loader):


    prior: str = dataclasses.field(
        default="uniform",
        doc="""Re-sampling mode for the discrete index.

    The option `empirical` uses label frequencies as they appear in the dataset.
    The option `uniform` re-samples the dataset and adjust the frequencies of less
    common class labels.
    For balanced datasets, it is typically more accurate to stick to the `empirical`
    option.
    """,
    )

    @property
    def index(self):
        """The (discrete) dataset index."""
        return self.dataset.discrete_index

    def __post_init__(self):
        super().__post_init__()
        if self.dataset.discrete_index is None:
            raise ValueError("Dataset does not provide a discrete index.")
        self._init_distribution()

    def _init_distribution(self):
        if self.prior == "uniform":
            self.distribution = CustomDiscreteUniform(
                self.index)
        elif self.prior == "empirical":
            self.distribution = CustomDiscreteEmpirical(
                self.index)
        else:
            raise ValueError(
                f"Invalid choice of prior distribution. Got '{self.prior}', but "
                f"only accept 'uniform' or 'empirical' as potential values.")

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference samples will be sampled from the empirical or uniform prior
        distribution (if uniform, the discrete index values will be used to perform
        histogram normalization).

        The positive samples will be sampled such that their discrete index value
        corresponds to the respective value of the reference samples.

        The negative samples will be sampled from the same distribution as the
        reference examples.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.
        """
        reference_idx = self.distribution.sample_prior(num_samples * 2)
        negative_idx = reference_idx[num_samples:]
        reference_idx = reference_idx[:num_samples]
        reference = self.index[reference_idx] # TODO: potentially change such that only movement is passed not coh + mov
        positive_idx = self.distribution.sample_conditional(reference)
        return BatchIndex(reference=reference_idx,
                          positive=positive_idx,
                          negative=negative_idx)

Loader = CustomCohortDiscreteDataLoader(dataset=Datasingle,num_steps=100,batch_size=25)

solver.fit(Loader)