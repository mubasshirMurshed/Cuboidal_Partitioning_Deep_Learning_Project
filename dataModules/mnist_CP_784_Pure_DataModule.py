from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from datasets import MNISTGraphDataset_V6
from typing import List, Union


class MNIST_CP_784_Pure_DataModule(DataModule):
    """
    A data module for the cuboidal graph dataset that has a pure cap of 64 across all images
    """
    def __init__(self, batch_size: int, caps: Union[List[int], None]=None):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        - caps: List[int]
            - Length caps for training and validation datasets
        """
        if caps is not None:
            self.caps = caps
        else:
            self.caps = [None, None]
        super().__init__(None, None, batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTGraphDataset_V6(root="data/mnist784/",
                                              name="mnistTrain",
                                              mode="CP",
                                              partition_limit=784,
                                              x_centre=False,
                                              y_centre=False,
                                              angle=False
                                             )

        self.val_set = MNISTGraphDataset_V6(root="data/mnist784/",
                                            name="mnistTest",
                                            mode="CP",
                                            partition_limit=784,
                                            x_centre=False,
                                            y_centre=False,
                                            angle=False
                                           )