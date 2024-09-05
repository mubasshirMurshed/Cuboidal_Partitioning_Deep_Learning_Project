import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GAT_Model(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        # Init parent
        super().__init__()

        # GCN layers
        self.initial_conv = GATv2Conv(in_channels=num_features, out_channels=64, heads=3)
        self.conv1 = GATv2Conv(in_channels=64*3, out_channels=64, heads=2)
        self.conv2 = GATv2Conv(in_channels=64*2, out_channels=64, heads=2, concat=False)

        # Output layer
        self.out = nn.Linear(64*2, num_classes)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out
