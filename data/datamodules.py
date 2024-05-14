from abc import abstractmethod
from torch.utils import data as std
from torch_geometric import loader as graph
from typing import Union
from torch_geometric.loader import DataLoader
from .datasets import GraphDataset_CSV


class DataModule():
    """
    A base class for data module encapsulation that allows easy instantiation of an organised
    class that handles dataset loading and dataloader set up.
    """
    def __init__(self, batch_size: int, dataloader_class: Union[std.DataLoader, graph.DataLoader]):
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
        self.batch_size = batch_size
        self.dataloader_class = dataloader_class
        self.train_set = None
        self.val_set = None
        self.test_set = None
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
    

class General_DataModule(DataModule):
    """
    A data module for the cuboidal graph dataset.
    """
    def __init__(self, dataset:str, num_segments: int, batch_size: int, mode: str, x_centre=False,
                y_centre=False, colour=False, num_pixels=False, angle=False,
                width=False, height=False, stdev=False):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        self.dataset_name = dataset
        self.num_segments = num_segments
        self.mode = mode
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.colour = colour
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height
        self.stdev = stdev
        super().__init__(batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = GraphDataset_CSV(  root="data/csv/",
                                            dataset=self.dataset_name,
                                            split="Train",
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            x_centre=self.x_centre,
                                            y_centre=self.y_centre,
                                            colour=self.colour,
                                            num_pixels=self.num_pixels,
                                            angle=self.angle,
                                            width=self.width,
                                            height=self.height,
                                            stdev=self.stdev
        )

        self.val_set = GraphDataset_CSV(    root="data/csv/",
                                            dataset=self.dataset_name,
                                            split="Validation",
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            x_centre=self.x_centre,
                                            y_centre=self.y_centre,
                                            colour=self.colour,
                                            num_pixels=self.num_pixels,
                                            angle=self.angle,
                                            width=self.width,
                                            height=self.height,
                                            stdev=self.stdev
        )
        
        self.test_set = GraphDataset_CSV(   root="data/csv/",
                                            dataset=self.dataset_name,
                                            split="Test",
                                            mode=self.mode,
                                            num_segments=self.num_segments,
                                            x_centre=self.x_centre,
                                            y_centre=self.y_centre,
                                            colour=self.colour,
                                            num_pixels=self.num_pixels,
                                            angle=self.angle,
                                            width=self.width,
                                            height=self.height,
                                            stdev=self.stdev
        )