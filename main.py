# %% Imports
import torch
from torch import nn
from datasets import *
from models import *
from utils.training import *
from utils.logger import *
from dataModules import *

# %% Seeding
torch.manual_seed(42)

# %% Instantiate model
num_classes = 10
num_cuboids = 128
# model = MNIST_Sparse(numCuboids=num_cuboids, numFeatures=5, numClasses=num_classes)
model = MNIST_CNN()

# Define hyperparameters
hparams = {
    "max_epochs" : 100,
    "learning_rate" : 0.002,
    "batch_size" : 128
}
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["learning_rate"])
hparams["optimizer"] = optimizer.__class__.__name__
hparams["loss_fn"] = loss_fn.__class__.__name__

# Define flags
allow_log = True
save_every_n_epoch = 10
train_root = "data/mnistCuboidalData/mnistTrain128_.csv"
val_root = "data/mnistCuboidalData/mnistTest128_.csv"
train_length = 2000
val_length = 500

# %% Create data module
# data_module = MNISTCuboidalSparseDataModule(train_root, val_root, hparams["batch_size"], num_cuboids, [train_length, val_length])
data_module = MNISTDataModule("data/mnistPytorch", "data/mnistPytorch", hparams["batch_size"])

# %% Running script
trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, optimizer=optimizer, hparams=hparams,
                save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes,)

if __name__ == "__main__":
    trainer.fit()