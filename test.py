# %% Imports
from dataModules.mnistGraphDataModule import MNISTGraphDataModule
from dataModules.mnistSuperpixelDataModule import MNISTSuperpixelDataModule

# %% Run data module
dm = MNISTGraphDataModule(10, [5000, 1000])
dm.setup()

# %%
dm = MNISTSuperpixelDataModule(10)
dm.setup()

# %%
