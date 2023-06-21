# %% Imports
import torch
from torch import nn
from datasets import *
from models import *
from training import *
from utils.logger import *
from dataModules import *

# %% Seeding
torch.manual_seed(42)

# %% Define hyperparameters
hparams = {
    "max_epochs" : 100,
    "learning_rate" : 0.001,
    "batch_size" : 64
}

# %% Create data module
# data_module = MNISTCuboidalSparseDataModule(train_root, val_root, hparams["batch_size"], num_cuboids, [train_length, val_length])
# data_module = MNISTDataModule("data/mnistPytorch", "data/mnistPytorch", hparams["batch_size"])
# data_module = MNISTSuperpixelDataModule(hparams["batch_size"])
# data_module = MNISTGraphDataModule(hparams["batch_size"], [5000, 1000])
# data_module = MNIST_SP_128_DataModule(hparams["batch_size"])
# data_module = MNIST_OR_16_Block_DataModule("data/mnistPytorch", "data/mnistPytorch", hparams["batch_size"])
# data_module = MNIST_RP_16_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_64_Pure_DataModule(hparams["batch_size"])
data_module = MNIST_CTP_64_Pure_DataModule(hparams["batch_size"])

# %% Instantiate model
num_classes = 10
# num_cuboids = 128
# model = MNIST_Sparse(numCuboids=num_cuboids, numFeatures=5, numClasses=num_classes)
# model = MNIST_CNN()
model = MNIST_GAT(num_features=data_module.train_set.num_features)
# model = MNIST_GAT2(num_features=data_module.train_set.num_features)
# model = MNIST_CNN_2()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["learning_rate"])
hparams["optimizer"] = optimizer.__class__.__name__
hparams["loss_fn"] = loss_fn.__class__.__name__

# Define flags
allow_log = True
save_every_n_epoch = 1
resume_from_ckpt= None
is_graph_model = True

# %% Running script
trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, optimizer=optimizer, hparams=hparams,
                save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes, is_graph_model=is_graph_model,
                resume_from_ckpt=resume_from_ckpt)

if __name__ == "__main__":
    trainer.fit()