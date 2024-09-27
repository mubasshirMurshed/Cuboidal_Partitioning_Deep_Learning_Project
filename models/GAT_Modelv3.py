import torch
from torch import nn
from torch_geometric.nn import DynamicEdgeConv
import torch.nn.functional as F
from torch_geometric.nn.aggr import MeanAggregation, StdAggregation, MaxAggregation

class GAT_Modelv3(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        # Init parent
        super().__init__()

        # GCN layers
        # self.initial_conv = GATv2Conv(in_channels=num_features, out_channels=64, heads=3)
        self.initial_conv = DynamicEdgeConv(nn=nn.Linear(2*num_features, 64*3), k=10)

        # self.conv1 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3)
        self.conv1 = DynamicEdgeConv(nn=nn.Linear(2*64*3, 64*3), k=10)
        # self.conv2 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3, concat=False)
        self.conv2 = DynamicEdgeConv(nn=nn.Linear(2*64*3, 64), k=10)

        # Pooling layers
        self.gmp = MaxAggregation()
        self.gstdp = StdAggregation()
        self.gap = MeanAggregation()

        # Output layer
        self.fc1 = nn.Linear(64*3, 64)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden)
        hidden = F.relu(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([self.gmp(x=hidden, index=batch_index), 
                            self.gap(x=hidden, index=batch_index),
                            self.gstdp(x=hidden, index=batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.fc1(hidden)
        out = self.dropout(out)
        out = self.fc2(out)
        return out