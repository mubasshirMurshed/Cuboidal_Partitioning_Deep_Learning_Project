from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from datasets import MNISTGraphDataset_CSV


class MNIST_CP_DataModule(DataModule):
    """
    A data module for the cuboidal graph dataset.
    """
    def __init__(self, num_cuboids: int, batch_size: int):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        self.num_cuboids = num_cuboids
        super().__init__(batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTGraphDataset_CSV(root="data/",
                                              split="Train",
                                              mode="CP",
                                              num_cuboids=self.num_cuboids,
                                              x_centre=True,
                                              y_centre=True,
                                              colour=True,
                                              num_pixels=True,
                                              angle=True
                                             )

        self.val_set = MNISTGraphDataset_CSV(root="data/",
                                            split="Validation",
                                            mode="CP",
                                            num_cuboids=self.num_cuboids,
                                            x_centre=True,
                                            y_centre=True,
                                            colour=True,
                                            num_pixels=True,
                                            angle=True
                                           )
        
        self.test_set = MNISTGraphDataset_CSV(root="data/",
                                            split="Test",
                                            mode="CP",
                                            num_cuboids=self.num_cuboids,
                                            x_centre=True,
                                            y_centre=True,
                                            colour=True,
                                            num_pixels=True,
                                            angle=True
                                           )