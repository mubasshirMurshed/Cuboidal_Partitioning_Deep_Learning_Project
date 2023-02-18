# %%
from dataModules import *
dm = MNIST_CTP_64_Pure_DataModule(batch_size=64)

# %%
# from datasets import *
# ds = MNISTGraphDataset_V5(root="data/mnist64/",
#                           name="mnistTrain",
#                           mode="CP",
#                           partition_limit=64,
#                           # y_centre=False,
#                         #   angle=False
#                           )

# %%
print("wow")