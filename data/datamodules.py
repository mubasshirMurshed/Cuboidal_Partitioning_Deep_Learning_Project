from torch.utils.data import DataLoader as TorchDataLoader
from typing import Dict, Type
from torch_geometric.loader import DataLoader as PyGDataLoader
from .transforms import CuPIDTransform, SLICTransform, CuPIDPartition, SLICPartition
from .graph_datasets import Graph_Dataset, Graph_Dataset_CSV
from .data_classes import SourceDataset
from enums import Split, Partition
import signal


def worker_init(x):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class DataModule():
    """
    A base class for data module encapsulation that allows easy instantiation of an organised
    class that handles dataset loading and dataloader set up.
    """
    def __init__(self, dataset: SourceDataset, batch_size: int, num_workers: int):
        """
        Save information as attributes.

        Args:
        - dataset: SourceDataset
            - The source dataset to be referred to
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        # Save attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = dataset.num_classes
        self.num_features = dataset.shape[-1]

        # Create references to the original datasets
        self.train_set = dataset.train_dataset()
        self.validation_set = dataset.validation_dataset()
        self.test_set = dataset.test_dataset()


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return TorchDataLoader(
            dataset=self.train_set, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )


    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return TorchDataLoader(
            dataset=self.validation_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )
    
    
    def test_dataloader(self):
        """
        Returns the testing dataloader.
        """
        return TorchDataLoader(
            dataset=self.train_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )
    

class Graph_DataModule_CSV(DataModule):
    """
    A data module for loading in graph data in CSV files using the GraphDataset_CSV standard.
    """
    def __init__(self, dataset: SourceDataset, num_segments: int, batch_size: int, mode: Partition, 
                 features: Dict[str, bool], num_workers: int=0):
        """
        Save attributes.

        Args:
        - dataset: SourceDataset
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
        # Save attributes
        super().__init__(dataset, batch_size, num_workers)
        self.num_segments = num_segments
        self.features = features
        self.mode = mode

        # Create references to the partitioned datasets
        if mode is Partition.CuPID:
            transform = CuPIDPartition(num_segments) 
        elif mode is Partition.SLIC:
            transform = SLICPartition(num_segments)
        else:
            raise ValueError(f"Supplied 'mode' argument not a registered partitioning strategy. Got {self.mode} but should be been a Partition Enum.")
        self.partition_train_set = self.dataset.train_dataset(transform)
        self.partition_validation_set = self.dataset.validation_dataset(transform)
        self.partition_test_set = self.dataset.test_dataset(transform)
        
        # Create references to the PyG graph in memory datasets
        self.graph_train_set = Graph_Dataset_CSV( 
            root="data/csv/",
            name=self.dataset.name,
            split=Split.TRAIN,
            mode=self.mode,
            num_segments=self.num_segments,
            **self.features
        )

        self.graph_validation_set = Graph_Dataset_CSV(
            root="data/csv/",
            name=self.dataset.name,
            split=Split.VALIDATION,
            mode=self.mode,
            num_segments=self.num_segments,
            **self.features
        )
        
        self.graph_test_set = Graph_Dataset_CSV(
            root="data/csv/",
            name=self.dataset.name,
            split=Split.TEST,
            mode=self.mode,
            num_segments=self.num_segments,
            **self.features
        )

        self.num_features = self.graph_train_set.num_features


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_train_set, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )


    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_validation_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )
    
    
    def test_dataloader(self):
        """
        Returns the testing dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )


class Graph_DataModule(DataModule):
    """
    A data module for loading in graph data and transforming on the fly.
    """
    def __init__(self, dataset: SourceDataset, num_segments: int, batch_size: int, mode: Partition,
                 features: Dict[str, bool], num_workers: int=0):
        """
        Save attributes.

        Args:
        - dataset: SourceDataset
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
        # Save attributes
        super().__init__(dataset, batch_size, num_workers)
        self.num_segments = num_segments
        self.features = features
        self.mode = mode

        # Create references to the partitioned datasets
        if mode is Partition.CuPID:
            transform = CuPIDTransform(num_segments)
            partition = CuPIDPartition(num_segments)
        elif mode is Partition.SLIC:
            transform = SLICTransform(num_segments)
            partition = SLICPartition(num_segments)
        else:
            raise ValueError(f"Supplied 'mode' argument not a registered partitioning strategy. Got {self.mode} but should be been a Partition Enum.")
        self.partition_train_set = self.dataset.train_dataset(transform)
        self.partition_validation_set = self.dataset.validation_dataset(transform)
        self.partition_test_set = self.dataset.test_dataset(transform)
        
        # Create references to the PyG graph datasets
        self.graph_train_set = Graph_Dataset(
            dataset=self.dataset.train_dataset(partition),
            num_segments=self.num_segments,
            **self.features
        )

        self.graph_validation_set = Graph_Dataset(
            dataset=self.dataset.validation_dataset(partition),
            num_segments=self.num_segments,
            **self.features
        )
        
        self.graph_test_set = Graph_Dataset(
            dataset=self.dataset.test_dataset(partition),
            num_segments=self.num_segments,
            **self.features
        )

        self.num_features = self.graph_train_set.num_features


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_train_set, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )


    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_validation_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )
    
    
    def test_dataloader(self):
        """
        Returns the testing dataloader.
        """
        return PyGDataLoader(
            dataset=self.graph_test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=worker_init
        )