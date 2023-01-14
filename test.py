# %% Imports
from dataModules.mnistGraphDataModule import MNISTGraphDataModule

# %% Run data module
dm = MNISTGraphDataModule(10, [5000, 1000])
dm.setup()

# %%
