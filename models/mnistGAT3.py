import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn.aggr import SoftmaxAggregation, PowerMeanAggregation, SetTransformerAggregation

class MNIST_GAT3(torch.nn.Module):
    def __init__(self, num_features: int):
        # Init parent
        super().__init__()

        # GCN layers
        self.initial_conv = GATv2Conv(in_channels=num_features, out_channels=128, heads=3)
        self.conv1 = GATv2Conv(in_channels=128*3, out_channels=128, heads=3)
        self.conv2 = GATv2Conv(in_channels=128*3, out_channels=64, heads=3)
        self.conv3 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3, concat=False)

        # Pooling layers
        # self.sma = SoftmaxAggregation(learn=True)
        # self.pma = PowerMeanAggregation(learn=True)
        self.sta = SetTransformerAggregation(channels=64, heads=2)

        # Output layer
        self.fc1 = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)

        # Global Pooling (stack different aggregations)
        # hidden = torch.cat([self.sma(x=hidden, index=batch_index),
        #                     self.pma(x=hidden, index=batch_index)], dim=1)
        
        hidden = torch.cat([self.sta(x=hidden, index=batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.fc1(hidden)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out