# %% Imports
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
from torch import nn
from data.datamodules import General_DataModule
from models.mnistGAT2 import MNIST_GAT2
from trainer import Trainer
import torch_geometric
# torch.backends.cudnn.deterministic=True
# torch.use_deterministic_algorithms(mode=True)

# %% Seeding
seed = 42
torch_geometric.seed_everything(seed)

# %% Define hyperparameters
hparams = {
    "max_epochs" : 100,
    "learning_rate" : 0.001,
    "batch_size" : 64
}

# %% Create data module
data_module = General_DataModule(
                        "mnist", 64, hparams["batch_size"], "CP",
                        x_centre=True,
                        y_centre=True,
                        width=True,
                        height=True,
                        colour=True,
                        num_pixels=True,
                        stdev=True)

# %% Instantiate model
num_classes = 10
model = MNIST_GAT2(num_features=data_module.train_set.num_features)
# model = torch.compile(model, dynamic=True)  # :(

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=hparams["learning_rate"])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
hparams["optimizer"] = optimizer.__class__.__name__
hparams["loss_fn"] = loss_fn.__class__.__name__
hparams["scheduler"] = scheduler.__class__.__name__

# Define flags
allow_log = True
save_every_n_epoch = 10
resume_from_ckpt = None
is_graph_model = True

# %% Running script
trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, hparams=hparams,
                save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes, is_graph_model=is_graph_model,
                resume_from_ckpt=resume_from_ckpt)

if __name__ == "__main__":
    trainer.fit()
    trainer.test()