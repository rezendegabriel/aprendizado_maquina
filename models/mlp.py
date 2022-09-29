#%% LIBRARIES

import torch.nn as nn
import torch.nn.functional as F

#%% CLASS

class MLPLayers(nn.Module):
    def __init__(self, num_inputs, layer_size, drop):
        super().__init__()

        self.lin1 = nn.Linear(num_inputs, num_inputs*layer_size)
        self.lin2 = nn.Linear(num_inputs*layer_size, num_inputs*layer_size)

        self.drop = nn.Dropout(p = drop)

    def forward(self, x):
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))

        return x

class MLP(MLPLayers):
    def __init__(self, num_inputs, num_outputs, layer_size, drop):
        super().__init__(num_inputs, layer_size, drop)

        self.lin3 = nn.Linear(num_inputs*layer_size, num_outputs)

    def forward(self, x):
        x = super().forward(x)

        x = self.lin3(x)
        
        return x