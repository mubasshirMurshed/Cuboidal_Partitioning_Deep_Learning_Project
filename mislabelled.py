# %% Imports
import torch
from torch import nn
from datasets import *
from models import *
from training import *
from utils.logger import *
from dataModules import *
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

# %% Seeding
torch.manual_seed(42)

# %% Define hyperparameters
hparams = {
    "max_epochs" : 100,
    "learning_rate" : 0.001,
    "batch_size" : 256
}

# %% Create data module
# data_module = MNIST_CP_16_Pure_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_32_Pure_DataModule(hparams["batch_size"])
data_module = MNIST_CP_64_Pure_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_784_Pure_DataModule(hparams["batch_size"])
# data_module = MNIST_CP_128_DataModule(hparams["batch_size"])

# %% Instantiate model
model = MNIST_GAT3(num_features=data_module.train_set.num_features)
# model = MNIST_GAT2(num_features=data_module.train_set.num_features)
# resume_from_ckpt=r"saved\MNIST_CP_16_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-29__12-05-22\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_32_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-29__01-57-53\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_64_Pure_DataModule\MNIST_GAT\Run_ID__2023-01-26__01-42-42\checkpoints\last.pt"
# resume_from_ckpt=r"saved\MNIST_CP_128_DataModule\MNIST_GAT\Run_ID__2023-01-20__19-44-53\checkpoints\epoch=12-val_loss=0.0698-val_acc=0.9825.pt"
resume_from_ckpt=r"saved\MNIST_CP_64_Pure_DataModule\XYCNA\MNIST_GAT3\Run_ID__2023-06-21__17-08-49\checkpoints\epoch=27-val_loss=0.1344-val_acc=0.9883.pt"
num_classes = 10
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# %% Running script
# Load in model
if resume_from_ckpt is not None:
    model.load_state_dict(torch.load(resume_from_ckpt))
    print("Model successfully loaded from " + resume_from_ckpt)
    print('-' * 20)

val_loader = data_module.val_dataloader()
model.eval() # puts the model in validation mode
running_loss = 0
acc1 = MulticlassAccuracy(num_classes, 1, "weighted").to(device)
acc2 = MulticlassAccuracy(num_classes, 2, "weighted").to(device)
acc3 = MulticlassAccuracy(num_classes, 3, "weighted").to(device)
acc4 = MulticlassAccuracy(num_classes, 4, "weighted").to(device)
acc5 = MulticlassAccuracy(num_classes, 5, "weighted").to(device)
cm = MulticlassConfusionMatrix(num_classes, normalize='none').to(device)

mislabelled = []
with torch.no_grad(): # save memory by not saving gradients which we don't need 
    indices = torch.arange(0, 64)
    for batch in tqdm(val_loader, leave=False):
        # Get batch of images and labels
        batch = batch.to(device) # put the data on the GPU
        targets = batch.y

        # Forward
        preds = model(batch.x, batch.edge_index, batch.batch)

        # Calculate loss
        val_loss = loss_fn(preds, targets)
        running_loss += val_loss.item()

        # Update accuracy and CM
        acc1.update(preds, targets)
        acc2.update(preds, targets)
        acc3.update(preds, targets)
        acc4.update(preds, targets)
        acc5.update(preds, targets)
        cm.update(preds, targets)

        # Find and add mislabelled

avg_loss = running_loss/len(val_loader)
print(f"Loss is: {avg_loss}")
a1 = acc1.compute()
print(f"Top 1 Accuracy is: {a1:.4%}")
a2 = acc2.compute()
print(f"Top 2 Accuracy is: {a2:.4%}")
a3 = acc3.compute()
print(f"Top 3 Accuracy is: {a3:.4%}")
a4 = acc4.compute()
print(f"Top 4 Accuracy is: {a4:.4%}")
a5 = acc5.compute()
print(f"Top 5 Accuracy is: {a5:.4%}")
conf_matrix = cm.compute()
conf_matrix = conf_matrix.to("cpu")
print(conf_matrix.numpy())
# %%
