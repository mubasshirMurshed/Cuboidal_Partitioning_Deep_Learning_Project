# %% Imports
import matplotlib.pyplot as plt
from dataModules.mnist_OR_16_DataModule import MNIST_OR_16_DataModule
from dataModules.mnist_OR_16_Block_DataModule import MNIST_OR_16_Block_DataModule
from dataModules.mnist_OR_DataModule import MNIST_OR_DataModule
from torchvision.datasets import MNIST
import numpy as np

# %% Set up dataset
dm_og = MNIST_OR_DataModule("data/mnistPytorch", "data/mnistPytorch", 10)
dm_16 = MNIST_OR_16_DataModule("data/mnistPytorch", "data/mnistPytorch", 10)
dm_16_ = MNIST_OR_16_Block_DataModule("data/mnistPytorch", "data/mnistPytorch", 10)
ds_og = dm_og.train_set
ds_16 = dm_16.train_set
ds_16_ = dm_16_.train_set

# %% Images to display
for idx in range(10):
    img = ds_og[idx][0]
    img_resized = ds_16[idx][0]
    img_sectioned = ds_16_[idx][0]

    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(img.permute(1, 2, 0), cmap="gray")
    axarr[1].imshow(img_resized.permute(1, 2, 0), cmap="gray")
    axarr[2].imshow(img_sectioned.permute(1, 2, 0), cmap="gray")
# %%
