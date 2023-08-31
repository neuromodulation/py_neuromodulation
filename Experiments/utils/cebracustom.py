from typing import Literal, Optional, Union
import numpy.typing as npt
import scipy.interpolate

import cebra.distributions.base as abc_
import abc
import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra.data as cebra_data
from cebra.data.datatypes import BatchIndex
class CustomConditionalDistribution(abc.ABC):
    """Mixin for all conditional distributions.

    Conditional distributions return a batch of indices, based on
    a given batch of indices. Indexing the dataset with these indices
    will return samples from the conditional distribution.
    """

    @abc.abstractmethod
    def sample_conditional_cohmov(self, query: torch.Tensor) -> torch.Tensor:
        """Return indices for the conditional distribution samples

        Args:
            query: Indices of reference samples

        Returns:
            A tensor of indices. Indexing the dataset with these
            indices will return samples from the desired conditional
            distribution.
        """
        raise NotImplementedError()
    @abc.abstractmethod
    def sample_conditional_mov(self, query: torch.Tensor) -> torch.Tensor:
        """Return indices for the conditional distribution samples

        Args:
            query: Indices of reference samples

        Returns:
            A tensor of indices. Indexing the dataset with these
            indices will return samples from the desired conditional
            distribution.
        """
        raise NotImplementedError()
class CohortDiscrete(CustomConditionalDistribution, abc_.HasGenerator):
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
        self.movsorted_idx = torch.from_numpy(np.argsort(self.samples[:,0])) # Multi dim version --> Sort first by 0th (cohort), then by movement 1st)
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

        # Do same but only for movement
        self.counts = np.bincount(self.samples[:, 1])
        self.cdfmov = np.zeros((len(self.counts) + 1,))
        self.cdfmov[1:] = np.cumsum(self.counts)
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

    def sample_conditional_cohmov(self, reference_index: torch.Tensor) -> torch.Tensor:
        """Draw samples conditional on template samples.

        Args:
            samples: batch of indices, typically drawn from a
                prior distribution. Conditional samples will match
                the values of these indices

        Returns:
            batch of indices, whose values match the values
            corresponding to the given indices.
        """
        reference_index = self._to_numpy_int(2*reference_index[:,0] + reference_index[:,1])
        idx = np.random.uniform(0, 1, len(reference_index))
        idx *= self.cdf[reference_index + 1] - self.cdf[reference_index]
        idx += self.cdf[reference_index]
        idx = idx.astype(int)

        return self.sorted_idx[idx]
    def sample_conditional_mov(self, reference_index: torch.Tensor) -> torch.Tensor:
        """Draw samples conditional on template samples.

        Args:
            samples: batch of indices, typically drawn from a
                prior distribution. Conditional samples will match
                the values of these indices

        Returns:
            batch of indices, whose values match the values
            corresponding to the given indices.
        """
        reference_index = self._to_numpy_int(reference_index[:,1])
        idx = np.random.uniform(0, 1, len(reference_index))
        idx *= self.cdfmov[reference_index + 1] - self.cdfmov[reference_index]
        idx += self.cdfmov[reference_index]
        idx = idx.astype(int)

        return self.movsorted_idx[idx]

class CohortDiscreteUniform(CohortDiscrete, abc_.PriorDistribution):
    """Re-sample the given indices and produce samples from a uniform distribution."""

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        return self.sample_uniform(num_samples)


class CohortDiscreteEmpirical(CohortDiscrete, abc_.PriorDistribution):
    """Draw samples from the empirical distribution defined by the passed index."""

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        return self.sample_empirical(num_samples)
@dataclasses.dataclass
class CohortDiscreteDataLoader(cebra_data.Loader):


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

    cond: str = dataclasses.field(
        default="cohmov",
        doc="""Which label do you want to match between the positive sample and the reference.
        
        Choice between: 'cohmov' to match both cohort and movement label (thus CEBRA will try to separate the cohorts)
                        and 'mov' to only match movement label and thus cohorts are not discerned
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
            self.distribution = CohortDiscreteUniform(
                self.index)
        elif self.prior == "empirical":
            self.distribution = CohortDiscreteEmpirical(
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
        reference = self.index[reference_idx]
        if self.cond == 'cohmov':
            positive_idx = self.distribution.sample_conditional_cohmov(reference)
        elif self.cond == 'mov':
            positive_idx = self.distribution.sample_conditional_mov(reference)
        return BatchIndex(reference=reference_idx,
                          positive=positive_idx,
                          negative=negative_idx)