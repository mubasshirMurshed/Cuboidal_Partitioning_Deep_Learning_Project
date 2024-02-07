# %% Imports
import torch
from torch import nn
from datasets import *
from models import *
from trainer import *
from utils.logger import *
from dataModules import *
import torch_geometric

# %% Seeding
seed = 42
torch_geometric.seed_everything(seed)

# %% Define hyperparameters
hparams = {
    "max_epochs" : 2,
    "learning_rate" : 0.001,
    "batch_size" : 64
}

# %% Create data module
data_module = MNIST_CP_64_Pure_DataModule(hparams["batch_size"])

# %% Instantiate model
num_classes = 10
model = MNIST_GAT2(num_features=data_module.train_set.num_features)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["learning_rate"])
hparams["optimizer"] = optimizer.__class__.__name__
hparams["loss_fn"] = loss_fn.__class__.__name__

# Define flags
allow_log = False
save_every_n_epoch = 1
resume_from_ckpt= None
is_graph_model = True

# %% Running script
trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, optimizer=optimizer, hparams=hparams,
                save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes, is_graph_model=is_graph_model,
                resume_from_ckpt=resume_from_ckpt)

if __name__ == "__main__":
    trainer.fit()