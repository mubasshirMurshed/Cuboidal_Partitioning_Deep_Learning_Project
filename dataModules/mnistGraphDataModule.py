from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from datasets import MNISTGraphDataset
from typing import List


class MNISTGraphDataModule(DataModule):
    """
    A data module for the cuboidal sparse dataset
    """
    def __init__(self, train_dir: str, val_dir: str, batch_size: int, caps: List[int]):
        """
        Save attributes.

        Args:
        - train_dir: str
            - Directory of training dataset
        - val_dir: str
            - Directory of validation dataset
        - batch_size: int
            - How many data samples per batch to be loaded
        - caps: List[int]
            - Length caps for training and validation datasets
        """
        super().__init__(train_dir, val_dir, batch_size, DataLoader)
        self.caps = caps
    

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTGraphDataset(root="data/mnistNew", filename="mnistTrain128", length=self.caps[0])

        self.val_set = MNISTGraphDataset(root="data/mnistNew", filename="mnistTest128", length=self.caps[1])