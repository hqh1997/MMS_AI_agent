import torch
import torch.nn as nn
from Networks.utils import init_max_weights

class ConcatFusion(nn.Module):
    def __init__(self, dims: list, hidden_size: int = 256, output_size: int = 256):
        super(ConcatFusion, self).__init__()
        self.dims = dims  
        self.hidden_size = hidden_size 
        self.output_size = output_size  
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(sum(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ])

    def forward(self, *x):
        concat = torch.cat(x, dim=0)
        return self.fusion_layer(concat)

class GatedConcatFusion(nn.Module):
    def __init__(self, dims: list, hidden_size: int = 256, output_size: int = 256):
        super(GatedConcatFusion, self).__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) for dim in dims])
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(sum(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ])

    def forward(self, *x):
      
        device = x[0].device
        x = [item.to(device) for item in x]

        items = []
        for gate, item in zip(self.gates, x):
            g = gate(item)
            items.append(item * g)
        concat = torch.cat(items, dim=0)  
        return self.fusion_layer(concat)
