from abc import abstractmethod
from torch.utils import data as std
from torch_geometric import loader as graph
from typing import Union, Dict, Type
from torch_geometric.loader import DataLoader
from .graph_datasets import Graph_Dataset_CSV
from data.data_classes import SourceDataset
from enums import Split, Partition

# TODO: Add k-fold functionality
# TODO: Fix num workers by putting main.py in a main() call

class DataModule():
    """
    A base class for data module encapsulation that allows easy instantiation of an organised
    class that handles dataset loading and dataloader set up.
    """
    def __init__(self, batch_size: int, dataset: Type[SourceDataset], mode: Partition, dataloader_class: Union[std.DataLoader, graph.DataLoader]):
        """
        Save information as attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        - dataset: Type[SourceDataset]
            - The source dataset to be referred to
        - mode: str
            - The mode of partitioning
        - dataloader_class: torch.utils.data.DataLoader | torch_geomteric.loader.DataLoader
            - Either a torch or torch_geometric dataloader
        """
        self.batch_size = batch_size
        self.dataloader_class = dataloader_class
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.source = dataset
        self.mode = mode
        self.setup()


    @abstractmethod
    def setup(self):
        pass


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return self.dataloader_class(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)


    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return self.dataloader_class(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    
    def test_dataloader(self):
        """
        Returns the testing dataloader.
        """
        return self.dataloader_class(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    

class Graph_DataModule(DataModule):
    """
    A data module for loading in graph data in CSV files using the GraphDataset_CSV standard.
    """
    def __init__(self, dataset: Type[SourceDataset], num_segments: int, batch_size: int, mode: Partition, 
                 features: Dict[str, bool]):
        """
        Save attributes.

        Args:
        - dataset: Type[SourceDataset]
            - The dataset to be referred to
        - num_segments: int
            - Number of segments to partition the data into
        - batch_size: int
            - How many data samples per batch to be loaded
        - mode: str
            - The mode of partitioning
        - features: Dict[str, bool]
            - A dictionary of features and whether they should be included or not in the
              in-memory dataset.
        """
        self.num_segments = num_segments
        self.features = features
        super().__init__(batch_size, dataset, mode, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training, validation and testing.
        """
        self.train_set = Graph_Dataset_CSV( root="data/csv/",
                                            name=self.source.name(),
                                            split=Split.TRAIN,
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            **self.features
        )

        self.val_set = Graph_Dataset_CSV(   root="data/csv/",
                                            name=self.source.name(),
                                            split=Split.VALIDATION,
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            **self.features
        )
        
        self.test_set = Graph_Dataset_CSV(  root="data/csv/",
                                            name=self.source.name(),
                                            split=Split.TEST,
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            **self.features
        )