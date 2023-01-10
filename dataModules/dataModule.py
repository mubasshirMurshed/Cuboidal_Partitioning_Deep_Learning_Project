from abc import abstractmethod
from torch.utils import data as std
from torch_geometric import loader as graph
from typing import Union

class DataModule():
    """
    A base class for data module encapsulation that allows easy instantiation of an organised
    class that handles dataset loading and dataloader set up.
    """
    def __init__(self, train_dir: str, val_dir: str, batch_size: int, dataloader_class: Union[std.DataLoader, graph.DataLoader]):
        """
        Save information as attributes.

        Args:
        - train_dir: str
            - Directory of training dataset
        - val_dir: str
            - Directory of validation dataset
        - batch_size: int
            - How many data samples per batch to be loaded
        - dataloader_class: torch.utils.data.DataLoader | torch_geomteric.loader.DataLoader
            - Either a torch or torch_geometric dataloader
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.dataloader_class = dataloader_class
        self.train_set = None
        self.val_set = None

    @abstractmethod
    def setup(self):
        pass

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return self.dataloader_class(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return self.dataloader_class(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)