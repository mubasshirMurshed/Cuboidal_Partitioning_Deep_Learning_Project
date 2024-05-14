import torch
from torch import nn
from .mnistGAT2 import MNIST_GAT2

# Create new model by instantiating the two GAT2 ones and load them (maybe freeze?)
class EnsembleModel2(torch.nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.model1 = MNIST_GAT2(num_features)
        self.model2 = MNIST_GAT2(num_features)
        self.model3 = MNIST_GAT2(num_features)
        self.model4 = MNIST_GAT2(num_features)

        # Load both models
        self.model1.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__15-26-10\checkpoints\epoch=58-val_loss=0.1310-val_acc=0.9835.pt"))
        self.model2.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__16-51-17\checkpoints\last.pt"))
        self.model3.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-19__19-54-31\checkpoints\epoch=3-val_loss=0.0783-val_acc=0.9796.pt"))
        self.model4.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-19__20-33-58\checkpoints\epoch=11-val_loss=0.0770-val_acc=0.9824.pt"))

        # Freeze the models
        # for p in self.first_model.parameters():
        #     p.requires_grad = False
        # for p in self.second_model.parameters():
        #     p.requires_grad = False

        # Define ensemble layers
        self.fc1 = nn.Linear(40, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 10)

    def forward(self, graph1, graph2, graph3, graph4):

        # Forward propogate into pre trained graphs
        out1 = self.model1(graph1.x.float(), graph1.edge_index, graph1.batch)
        out2 = self.model2(graph2.x.float(), graph2.edge_index, graph2.batch)
        out3 = self.model2(graph3.x.float(), graph3.edge_index, graph3.batch)
        out4 = self.model2(graph4.x.float(), graph4.edge_index, graph4.batch)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        return out