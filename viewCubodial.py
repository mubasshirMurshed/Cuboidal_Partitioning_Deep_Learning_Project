# %% Imports
import matplotlib.pyplot as plt
from datasets.mnistSparseDataset import MNISTSparseDataset
from torchvision.datasets import MNIST
import numpy as np

# %% Set up dataset
data_root = r"data\archive\mnistTest64.csv"
ds_cuboid = MNISTSparseDataset(csv_file_dir=data_root, n=64, length=100)
ds_normal = MNIST(root=r"data\mnistPytorch", train=False)

# %% Images to display
idx = 3    #92
img = ds_normal[idx][0]         # PIL image of actual MNIST digit
cuboidal = ds_cuboid[idx][0]    # Cuboidal data of image

# Convert to numpy array
cuboidal = cuboidal.cpu().detach().numpy()

# Denormalize and convert to ints
cuboidal[:, :4] *= 28
cuboidal[:, 4] *= 255
cuboidal = cuboidal.astype(int)

# %% Create cuboidal space
img_cuboidal = np.zeros((28, 28))

# %% Construct cuboidal image
# Iterate over each cuboid
for block_features in cuboidal:
    # Each block feature has 5 elements [top left x, top left y, height, width, colour]
    x_pos = block_features[0]
    y_pos = block_features[1]
    width = block_features[2]
    height = block_features[3]
    colour = block_features[4]  # Greyscale colour needs to be 0-255
    img_cuboidal[y_pos : (y_pos + height), x_pos : (x_pos + width)] = colour

# %% Plot image
f, axarr = plt.subplots(1,2)
axarr[0].imshow(img, cmap="gray")
axarr[1].imshow(img_cuboidal, cmap="gray", vmin=0, vmax=255)
# %%
