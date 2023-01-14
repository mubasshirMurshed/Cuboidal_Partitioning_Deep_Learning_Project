from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MNISTSuperpixels


class MNISTSuperpixelDataModule(DataModule):
    """
    A data module for the superpixel MNIST dataset
    """
    def __init__(self, batch_size: int):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        super().__init__(None, None, batch_size, DataLoader)
    

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTSuperpixels(root="data/mnistSuperpixel", train=True)

        self.val_set = MNISTSuperpixels(root="data/mnistSuperpixel", train=False)