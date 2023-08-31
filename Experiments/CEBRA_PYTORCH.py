from torch import nn
import numpy as np
import torch
import cebra.data as cebra_data
import cebra.distributions
from Experiments.utils.cebracustom import CohortDiscreteDataLoader
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin, cebra_layers
@cebra.models.register("offset9-model") # --> add that line to register the model!
class MyModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 2), nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(4, 5)


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
        name="offset9-model",
        num_neurons=dataset.input_dimension,
        num_units=32,
        num_output=8,
    ) for dataset in Data.iter_sessions()]).to("cuda")

modelsingle = cebra.models.init(
        name="offset10-model",
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

Loader = CohortDiscreteDataLoader(dataset=Datasingle,num_steps=100,batch_size=25)

solver.fit(Loader)
batches = np.lib.stride_tricks.sliding_window_view(session1,9,axis=0)
embedding = solver.transform(torch.from_numpy(batches[:]).to('cuda')).to('cuda')