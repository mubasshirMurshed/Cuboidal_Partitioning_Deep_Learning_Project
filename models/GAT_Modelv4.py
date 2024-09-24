import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, BatchNorm
import torch.nn.functional as F
from torch_geometric.nn.aggr import MultiAggregation, SoftmaxAggregation

class GAT_Modelv4(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        # Init parent
        super().__init__()

        # Graph Layers
        self.conv1 = GATv2Conv(in_channels=num_features, out_channels=64, heads=3)
        self.norm1 = BatchNorm(in_channels=64*3)

        self.conv2 = GATv2Conv(in_channels=64*3, out_channels=128, heads=3)
        self.norm2 = BatchNorm(in_channels=128*3)

        self.conv3 = GATv2Conv(in_channels=128*3, out_channels=128, heads=3)
        self.norm3 = BatchNorm(in_channels=128*3)
        
        self.conv4 = GATv2Conv(in_channels=128*3, out_channels=128, heads=3)
        self.norm4 = BatchNorm(in_channels=128*3)

        self.conv5 = GATv2Conv(in_channels=128*3, out_channels=128, heads=3, concat=False)
        self.norm5 = BatchNorm(in_channels=128)

        # Readout Aggregation
        self.readout = MultiAggregation(["mean", "std", "max", SoftmaxAggregation(learn=True)])

        # Output layer
        self.fc1 = nn.Linear(128*4, 128)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(64, num_classes)


    def forward(self, x, edge_index, batch_index):
        # Apply conv + norm + relu layers
        out1 = self.conv1(x, edge_index)
        out1 = self.norm1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = self.norm2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = self.norm3(out3)
        out3 = F.relu(out3)

        out4 = self.conv4(out3, edge_index)
        out4 = self.norm4(out4)
        out4 = F.relu(out4)

        out4 = out2 + out4  # Skip connection

        out5 = self.conv5(out4, edge_index)
        out5 = self.norm5(out5)
        out5 = F.relu(out5)

        # Readout of graph into single embedding vector
        out = self.readout(out5, batch_index)

        # Apply a final MLP
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
