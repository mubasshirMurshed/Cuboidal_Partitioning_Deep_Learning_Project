# %% Imports
from dataModules.mnistGraphDataModule import MNISTGraphDataModule

# %% Run data module
dm = MNISTGraphDataModule(10, [100, 100])
dm.setup()

# %%
