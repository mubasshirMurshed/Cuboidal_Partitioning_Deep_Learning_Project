import torch
from torch import nn
from .GAT_Modelv2 import GAT_Modelv2


class Ensemble_Model(torch.nn.Module):
    def __init__(self, num_features: list[int], num_classes: int) -> None:
        super().__init__()

        # Create each submodel
        self.num_submodels = len(num_features)
        self.models = []
        for i in range(self.num_submodels):
            self.models.append(
                GAT_Modelv2(num_features[i], num_classes)
            )
        self.models = nn.ModuleList(self.models)

        # Load both models
        # self.model1.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__15-26-10\checkpoints\epoch=58-val_loss=0.1310-val_acc=0.9835.pt"))
        # self.model2.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-18__16-51-17\checkpoints\last.pt"))
        # self.model3.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-19__19-54-31\checkpoints\epoch=3-val_loss=0.0783-val_acc=0.9796.pt"))
        # self.model4.load_state_dict(torch.load(r"saved\MNIST_CP_DataModule\XYCNA\MNIST_GAT2\Run_ID__2024-02-19__20-33-58\checkpoints\epoch=11-val_loss=0.0770-val_acc=0.9824.pt"))

        # Freeze the models
        # for p in self.first_model.parameters():
        #     p.requires_grad = False
        # for p in self.second_model.parameters():
        #     p.requires_grad = False

        # Define ensemble layers
        self.fc1 = nn.Linear(self.num_submodels * num_classes, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
        

    def forward(self, *graphs):

        # Forward propogate into each submodel
        results = [0]*self.num_submodels
        for i in range(self.num_submodels):
            results[i] = self.models[i](graphs[i].x.float(), graphs[i].edge_index, graphs[i].batch)

        # Concatenate results of each submodel
        out = torch.cat(results, dim=1)

        # Pass into meta classifier
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        return out