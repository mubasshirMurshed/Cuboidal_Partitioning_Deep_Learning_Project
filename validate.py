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
# data_module = MNIST_CP_16_Pure_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_32_Pure_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_64_Pure_DataModule(hparams["batch_size"])
data_module = MNIST_CP_128_DataModule(hparams["batch_size"])

# %% Instantiate model
# model = MNIST_GAT(num_features=data_module.train_set.num_features)
model = MNIST_GAT2(num_features=data_module.train_set.num_features)
# resume_from_ckpt=r"saved\MNIST_CP_16_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-29__12-05-22\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_32_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-29__01-57-53\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_64_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-26__01-42-42\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_128_DataModule\MNIST_GAT\Run_ID__2023-01-20__19-44-53\checkpoints\epoch=12-val_loss=0.0698-val_acc=0.9825.pt"
resume_from_ckpt=None

# %% Running script
trainer = Trainer(model=model, data_module=data_module, loss_fn= nn.CrossEntropyLoss(), optimizer=None, hparams=hparams,
                save_every_n_epoch=1, allow_log=False, num_classes=10, is_graph_model=True,
                resume_from_ckpt=resume_from_ckpt)

if __name__ == "__main__":
    _, acc = trainer.validation_G()
    print(f"The accuracy of the model is {acc*100:.2f}%")
    trainer.generateConfusionMatrix(trainer.validation_G)
# %%
