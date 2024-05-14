import torch
from torch import nn
from .mnistGAT2 import MNIST_GAT2

# Create new model by instantiating the two GAT2 ones and load them (maybe freeze?)
class EnsembleModel(torch.nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.first_model = MNIST_GAT2(num_features)
        self.second_model = MNIST_GAT2(num_features)

        # Load both models
        self.first_model.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__15-26-10\checkpoints\epoch=58-val_loss=0.1310-val_acc=0.9835.pt"))
        self.second_model.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__16-51-17\checkpoints\last.pt"))

        # Freeze the models
        # for p in self.first_model.parameters():
        #     p.requires_grad = False
        # for p in self.second_model.parameters():
        #     p.requires_grad = False

        # Define ensemble layers
        self.fc1 = nn.Linear(20, 15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, 10)

    def forward(self, graph1, graph2):

        # Forward propogate into pre trained graphs
        out1 = self.first_model(graph1.x.float(), graph1.edge_index, graph1.batch)
        out2 = self.second_model(graph2.x.float(), graph2.edge_index, graph2.batch)

        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out