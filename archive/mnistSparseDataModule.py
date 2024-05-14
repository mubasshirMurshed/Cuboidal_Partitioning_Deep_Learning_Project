from dataModules.dataModule import DataModule
from torch.utils.data import DataLoader
from datasets import MNISTSparseDataset
from typing import List


class MNISTSparseDataModule(DataModule):
    """
    A data module for the cuboidal sparse dataset
    """
    def __init__(self, train_dir: str, val_dir: str, batch_size: int, nCuboids: int, caps: List[int]):
        """
        Save attributes.

        Args:
        - train_dir: str
            - Directory of training dataset
        - val_dir: str
            - Directory of validation dataset
        - batch_size: int
            - How many data samples per batch to be loaded
        - nCuboids: int
            - Number of cuboids in partition
        - caps: List[int]
            - Length caps for training and validation datasets
        """
        self.caps = caps
        self.nCuboids = nCuboids
        super().__init__(train_dir, val_dir, batch_size, DataLoader)
    

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTSparseDataset(csv_file_dir=self.train_dir, 
                                                    n=self.nCuboids, 
                                                    length=self.caps[0])

        self.val_set = MNISTSparseDataset(csv_file_dir=self.val_dir, 
                                                    n=self.nCuboids, 
                                                    length=self.caps[1])