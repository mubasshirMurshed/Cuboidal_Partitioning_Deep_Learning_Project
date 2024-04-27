from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from datasets.groupedDataset import GroupDataset2


class Group_DataModule2(DataModule):
    """
    A data module for the cuboidal graph dataset.
    """
    def __init__(self, batch_size):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        super().__init__(batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = GroupDataset2(root="data/", split="Train")

        self.val_set = GroupDataset2(root="data/", split="Validation")
        
        self.test_set = GroupDataset2(root="data/", split="Test")