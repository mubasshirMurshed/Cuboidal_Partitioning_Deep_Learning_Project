import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, BatchNorm
import torch.nn.functional as F
from torch_geometric.nn.aggr import MultiAggregation

class GAT_Modelv5_Regression(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        # Init parent
        super().__init__()

        # Graph Layers
        self.conv1 = GATv2Conv(in_channels=num_features, out_channels=64, heads=3)
        self.norm1 = BatchNorm(in_channels=64*3)

        self.conv2 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3)
        self.norm2 = BatchNorm(in_channels=64*3)

        self.conv3 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3)
        self.norm3 = BatchNorm(in_channels=64*3)
        
        self.conv4 = GATv2Conv(in_channels=64*3, out_channels=64, heads=3, concat=False)
        self.norm4 = BatchNorm(in_channels=64)

        # Readout Aggregation
        self.readout = MultiAggregation(["mean", "std", "max"])

        # Output layer
        self.fc1 = nn.Linear(64*3, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, x, edge_index, batch_index):
        # Apply conv + norm + relu layers
        out1 = self.conv1(x, edge_index)
        out1 = self.norm1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = self.norm2(out2)
        out2 = F.dropout(out2, training=self.training, p=0.2)
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = self.norm3(out3)
        out3 = F.relu(out3)

        out3 = out3 + out1  # Skip connection

        out4 = self.conv4(out3, edge_index)
        out4 = self.norm4(out4)
        out4 = F.relu(out4)

        # Readout of graph into single embedding vector
        out = self.readout(out4, batch_index)

        # Apply a final MLP
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
